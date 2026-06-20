#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
// #include <omp.h>
#include <deque>
// #include <chrono>
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6

    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;

    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标
    vector<int> max_indices;

    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;

    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };

    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };

    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };

    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // unordered_map: 无序映射
    int total_preterm = 0;

    vector<PT> preterminals;

    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;

    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
class PriorityQueue
{
public:

    // 原始版本
    vector<PT> priority;

    // MultiQueues优化版本
    // deque<PT> *priority_queues;

    // omp_lock_t *queue_locks;

    // int num_queues = 8;

    // 模型作为成员
    model m;
    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列初始化
    void init();

    // 根据PT生成猜测
    void Generate(PT pt);

    // 处理队首PT
    void PopNext();

    // 判断所有队列是否为空
    bool Empty();
    int total_guesses = 0;
    vector<string> guesses;
};