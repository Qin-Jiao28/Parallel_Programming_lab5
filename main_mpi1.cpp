#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程号
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取总进程数
    // ==================== 所有进程共同的准备工作 ====================
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    PriorityQueue q;
    q.mpi_rank = rank;     // 将当前的进程号同步给优先队列
    q.mpi_size = size;     // 将总进程数同步给优先队列

    unordered_set<string> test_set;
    int cracked = 0;        // Master 用于最终汇总的总数
    int local_cracked = 0;  // 每个 Worker 本地统计的破解数

    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string test_pw;
    while (test_data >> test_pw)
    {   
        test_count += 1;
        test_set.insert(test_pw);
        if (test_count >= 1000000) { break; }
    }

    // 所有进程在后台同时跑训练，但只有 rank 0 会打印进度
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank); // <--- 传入了 rank
    q.m.order(rank);
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    // ==================== 根据 Rank 进行核心业务分流 ====================
    // ==================== 根据 Rank 进行核心业务分流 ====================
    if (rank == 0) 
    {
        // ==================== MASTER 节点逻辑 ====================
        uint32x4_t a = vdupq_n_u32(1);
        uint32x4_t b = vdupq_n_u32(2);
        uint32x4_t c = vaddq_u32(a, b);
        uint32_t result[4];
        vst1q_u32(result, c);
        cout << result[0] << endl;
        
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
        string test_hashes[8] = {
            "e10adc3949ba59abbe56e057f20f883e", "5f4dcc3b5aa765d61d8327deb882cf99",
            "25d55ad283aa400af464c76d713c07ad", "d8578edf8458ce06fbc5bb76a58c5ca4",
            "25f9e794323b453885f5181f1b624d0b", "827ccb0eea8a706c4c34a16891f84e7b",
            "81dc9bdb52d04dc20036dbd8313ed055", "96e79218965eb72c92a549dd5a330112"
        };
        for (int i = 0; i < 8; i++) {
            bit32 state[4];
            MD5Hash(test_pws[i], state);
            stringstream ss;
            for (int i1 = 0; i1 < 4; i1 += 1) ss << std::setw(8) << std::setfill('0') << hex << state[i1];
            if (ss.str() != test_hashes[i]) { return 1; }
        }
        cout << "MD5Hash test passed!" << endl; //请不要修改这一行

        cout << "here" << endl;
        auto start = system_clock::now();

        int num_workers = size - 1;
        long long global_guesses_dispatched = 0; 
        int packet[2000]; 

        while (true)
        {
            if (q.priority.empty() || global_guesses_dispatched > 10000000)
            {
                packet[0] = -1; 
                for (int w = 1; w <= num_workers; ++w) {
                    MPI_Send(packet, 1, MPI_INT, w, 1, MPI_COMM_WORLD);
                }
                break;
            }

            vector<PT> round_new_pts;

            for (int w = 1; w <= num_workers; ++w)
            {
                if (!q.priority.empty() && global_guesses_dispatched <= 10000000)
                {
                    int batch_size = 0;
                    int idx = 1; 

                    while (batch_size < 5 && !q.priority.empty() && global_guesses_dispatched <= 10000000)
                    {
                        PT pt = q.priority.front();

                        packet[idx++] = pt.template_id;
                        packet[idx++] = pt.curr_indices.size();
                        for (size_t i = 0; i < pt.curr_indices.size(); ++i) {
                            packet[idx++] = pt.curr_indices[i];
                        }

                        global_guesses_dispatched += pt.max_indices.back();
                        batch_size++;

                        vector<PT> new_pts = pt.NewPTs();
                        round_new_pts.insert(round_new_pts.end(), new_pts.begin(), new_pts.end());

                        q.priority.erase(q.priority.begin());
                    }

                    packet[0] = batch_size; 
                    MPI_Send(packet, idx, MPI_INT, w, 1, MPI_COMM_WORLD);
                }
                else
                {
                    packet[0] = -2;
                    MPI_Send(packet, 1, MPI_INT, w, 1, MPI_COMM_WORLD);
                }
            }

            // 修正处 1：这里加上了完美的 MPI_STATUS_IGNORE
            for (int w = 1; w <= num_workers; ++w)
            {
                int confirm_rank;
                MPI_Recv(&confirm_rank, 1, MPI_INT, w, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            for (PT n_pt : round_new_pts)
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
        }

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        double master_local_hash = 0;
        MPI_Reduce(&master_local_hash, &time_hash, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_cracked, &cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // ==================== 原样保留的系统评测行 ====================
        cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;//请不要修改这一行
        cout << "Hash time:" << time_hash << "seconds"<<endl;//请不要修改这一行
        cout << "Train time:" << time_train <<"seconds"<<endl;//请不要修改这一行
        cout << "Cracked:" << cracked << endl;
    } 
    else 
    {
        // ==================== WORKER 节点逻辑 ====================
        double local_time_hash = 0;
        int packet[2000];

        while (true)
        {
            // 修正处 2：这里同样换成了标准的 MPI_STATUS_IGNORE
            MPI_Recv(packet, 2000, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int batch_size = packet[0];
            if (batch_size == -1) {
                break; 
            }
            
            if (batch_size == -2) {
                int my_rank = rank;
                MPI_Send(&my_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                continue;
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

            if (q.guesses.size() >= 1000000)
            {
                for (const string &guess : q.guesses) {
                    if (test_set.find(guess) != test_set.end()) { local_cracked += 1; }
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

            int my_rank = rank;
            MPI_Send(&my_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        if (!q.guesses.empty())
        {
            for (const string &guess : q.guesses) {
                if (test_set.find(guess) != test_set.end()) { local_cracked += 1; }
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

        // 修正处 3：删除了那行突兀的非编译伪代码宏
        MPI_Reduce(&local_time_hash, &time_hash, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_cracked, &cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
    
}
