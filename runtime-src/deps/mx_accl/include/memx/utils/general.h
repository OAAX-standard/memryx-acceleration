#ifndef MX_UTILS
#define MX_UTILS

#include <condition_variable>
#include <mutex>
#include <queue>

using namespace std;

namespace MX
{
    namespace Utils
    {
        template <typename T>
        class fifo_queue
        {
        private:
            std::queue<T> m_queue;
            std::mutex m_mutex;

        public:
            size_t size()
            {
                unique_lock<mutex> lock(m_mutex);
                return m_queue.size();
            }
            void push(T item)
            {
                unique_lock<mutex> lock(m_mutex);
                m_queue.push(item);
            }
            T pop()
            {
                unique_lock<mutex> lock(m_mutex);
                T item = m_queue.front();
                m_queue.pop();
                return item;
            }
            fifo_queue &operator=(const fifo_queue &rhs) // copy assignment
            {
                if (this == &rhs)
                {
                    return *this;
                }
            }
        };
    } // namespace Utils
} // namespace MX

#endif