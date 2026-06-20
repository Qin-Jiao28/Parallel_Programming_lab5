#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    // 1. 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ==================== 所有进程共同的准备工作 ====================
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    PriorityQueue q;
    q.mpi_rank = rank;     
    q.mpi_size = size;     

    // 每一个进程都在后台默默训练模型（只有 rank 0 打印进度）
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank); 
    q.m.order(rank); 
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init(); 

    // 加载正确性验证的目标数据集
    unordered_set<string> test_set;
    int cracked = 0;        // Master 用于最终汇总的总破解数
    int local_cracked = 0;  // 每个 Worker 本地统计的破解数

    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string test_pw;
    while (test_data >> test_pw)
    {   
        test_count += 1;
        test_set.insert(test_pw);
        if (test_count >= 1000000) { 
            break; 
        }
    }

    // ==================== 根据 Rank 进行核心业务分流 ====================
    if (rank == 0) 
    {
        // ==================== MASTER 节点逻辑 ====================
        cout << "here" << endl;
        auto start = system_clock::now();

        int active_workers = size - 1; 
        long long global_guesses_dispatched = 0; 
        int packet[2000]; 

        while (active_workers > 0)
        {
            int worker_rank;
            // 监听哪个 Worker 闲下来了
            MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 限制猜测上限为 10,000,000
            if (q.priority.empty() || global_guesses_dispatched > 10000000)
            {
                packet[0] = -1; // 终止令牌
                MPI_Send(packet, 1, MPI_INT, worker_rank, 1, MPI_COMM_WORLD);
                active_workers--;
            }
            else
            {
                int batch_size = 0;
                int idx = 1; 

                // 批量打包 50 个任务
                while (batch_size < 50 && !q.priority.empty() && global_guesses_dispatched <= 10000000)
                {
                    PT pt = q.priority.front();

                    packet[idx++] = pt.template_id;
                    packet[idx++] = pt.curr_indices.size();
                    for (size_t i = 0; i < pt.curr_indices.size(); ++i) {
                        packet[idx++] = pt.curr_indices[i];
                    }

                    global_guesses_dispatched += pt.max_indices.back();
                    batch_size++;

                    // 演进优先队列
                    vector<PT> new_pts = q.priority.front().NewPTs();
                    for (PT n_pt : new_pts)
                    {
                        q.CalProb(n_pt);
                        for (auto iter = q.priority.begin(); iter != q.priority.end(); iter++)
                        {
                            if (iter != q.priority.end() - 1 && iter != q.priority.begin())
                            {
                                if (n_pt.prob <= iter->prob && n_pt.prob > (iter + 1)->prob)
                                {
                                    q.priority.emplace(iter + 1, n_pt);
                                    break;
                                }
                            }
                            if (iter == q.priority.end() - 1)
                            {
                                q.priority.emplace_back(n_pt);
                                break;
                            }
                            if (iter == q.priority.begin() && iter->prob < n_pt.prob)
                            {
                                q.priority.emplace(iter, n_pt);
                                break;
                            }
                        }
                    }
                    q.priority.erase(q.priority.begin());
                }

                packet[0] = batch_size; 
                MPI_Send(packet, idx, MPI_INT, worker_rank, 1, MPI_COMM_WORLD);
            }
        }

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        // 规约汇聚：Master 自身没计算，传入 0 或本地变量
        double master_local_hash = 0;
        MPI_Reduce(&master_local_hash, &time_hash, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_cracked, &cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // ==================== 严格匹配 correctness_guess.cpp 的输出 ====================
        cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
        cout << "Hash time:" << time_hash << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;
        cout << "Cracked:" << cracked << endl;
    } 
    else 
    {
        // ==================== WORKER 节点逻辑 ====================
        double local_time_hash = 0;
        int packet[2000];

        while (true)
        {
            int my_rank = rank;
            MPI_Send(&my_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(packet, 2000, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int batch_size = packet[0];
            if (batch_size == -1) {
                break; 
            }

            int idx = 1;
            for (int b = 0; b < batch_size; ++b)
            {
                int template_id = packet[idx++];
                int num_indices = packet[idx++];

                PT pt = q.pt_templates[template_id];
                pt.curr_indices.clear();
                for (int i = 0; i < num_indices; ++i) {
                    pt.curr_indices.push_back(packet[idx++]);
                }

                q.Generate(pt);
            }

            // 内存阈值控制：满 1,000,000 条进行匹配与哈希
            if (q.guesses.size() >= 1000000)
            {
                // 先进行本地正确性命中比对
                for (const string &guess : q.guesses) {
                    if (test_set.find(guess) != test_set.end()) {
                        local_cracked += 1;
                    }
                }

                auto start_hash = system_clock::now();
                for (size_t i = 0; i + 3 < q.guesses.size(); i += 4)
                {
                    string inputs[4] = { q.guesses[i], q.guesses[i + 1], q.guesses[i + 2], q.guesses[i + 3] };
                    bit32 states[4][4];
                    MD5Hash_SIMD(inputs, states);
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                local_time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                q.guesses.clear(); 
            }
        }

        // 处理尾部残余口令
        if (!q.guesses.empty())
        {
            for (const string &guess : q.guesses) {
                if (test_set.find(guess) != test_set.end()) {
                    local_cracked += 1;
                }
            }

            auto start_hash = system_clock::now();
            for (size_t i = 0; i + 3 < q.guesses.size(); i += 4)
            {
                string inputs[4] = { q.guesses[i], q.guesses[i + 1], q.guesses[i + 2], q.guesses[i + 3] };
                bit32 states[4][4];
                MD5Hash_SIMD(inputs, states);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            local_time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            q.guesses.clear();
        }

        // 向 Master 节点规约投递本地统计结果
        MPI_Reduce(&local_time_hash, &time_hash, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_cracked, &cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // 3. 退出 MPI 环境
    MPI_Finalize();
    return 0;
}