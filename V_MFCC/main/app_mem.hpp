#pragma once

#include "esp_timer.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

namespace app
{
    class AppMem
    {
    private: 
        constexpr static const char *TAG{"app-mem"};
        esp_timer_handle_t m_periodic_timer;

        static void periodic_timer_callback(void *arg);

    public:
        void monitor(void);
        void print(void);
    };
}