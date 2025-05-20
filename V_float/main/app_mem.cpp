#include "app_mem.hpp"


namespace app
{
    void AppMem::periodic_timer_callback(void *arg)
    {
        ESP_LOGI(TAG, "------- mem stats -------");
        ESP_LOGI(TAG, "internal\t: %10u (free) / %10u (total)", 
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL), 
                 heap_caps_get_total_size(MALLOC_CAP_INTERNAL));

        ESP_LOGI(TAG, "spiram\t: %10u (free) / %10u (total)", 
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM), 
                 heap_caps_get_total_size(MALLOC_CAP_SPIRAM));
    }

    void AppMem::monitor(void)
    {
        const esp_timer_create_args_t periodic_timer_args = {
            .callback = periodic_timer_callback,
            .arg = this
        };

        ESP_ERROR_CHECK(esp_timer_create(&periodic_timer_args, &m_periodic_timer));
        ESP_ERROR_CHECK(esp_timer_start_periodic(m_periodic_timer, 5000000u));
    }

    void AppMem::print(void)
    {
        periodic_timer_callback(nullptr);
    }
}
