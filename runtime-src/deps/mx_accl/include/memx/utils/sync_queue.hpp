#ifndef SYNC_QUEUE_HPP
#define SYNC_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <optional>

using namespace std::literals;

template <typename T>
class sync_queue {
    public:
        sync_queue<T>(): m_max_size(0) {}
        sync_queue<T>(size_t max_size): m_max_size(max_size) {};
        bool push(const T& item, std::chrono::milliseconds timeout=0ms);
        std::optional<T> pop(std::chrono::milliseconds timeout=0ms);
        int get_size();
    private:
        size_t m_max_size{0};
        std::queue<T> m_queue;
        std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_end = false;
};

template <typename T>
bool sync_queue<T>::push(const T& item, std::chrono::milliseconds timeout) {
    if (m_max_size > 0)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (timeout > 0ms) {
            auto now = std::chrono::steady_clock::now();
            if (!m_cv.wait_until(lock, now + timeout, [this]() { return m_queue.size() < m_max_size; })) {
                return false;
            }
        }
        else {
            m_cv.wait(lock, [this]() { return m_queue.size() < m_max_size; });
        }
    }
    {
        std::lock_guard lock(m_mutex);
        m_queue.push(item);
    }
    m_cv.notify_one(); // TODO this can cause the same thread to be notified each time
    return true;
}

template <typename T>
std::optional<T> sync_queue<T>::pop(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (timeout > 0ms) {
        auto now = std::chrono::steady_clock::now();
        if (!m_cv.wait_until(lock, now + timeout, [this](){ return m_queue.size() > 0; })) {
            lock.unlock();
            return {};
        }
    }
    else {
        m_cv.wait(lock, [this](){ return (m_queue.size() > 0); });
    }
    T val{std::move(m_queue.front())};
    m_queue.pop();
    lock.unlock();
    m_cv.notify_one(); // TODO this can cause the same thread to be notified each time
    return val;
}

template <typename T>
int sync_queue<T>::get_size(){
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.size();
}

#endif