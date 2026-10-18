idf_component_register(SRCS "main.c" "stream_core.c" "audio_frontend.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_gpio esp_wifi esp_event esp_netif esp_timer
             nvs_flash json esp_websocket_client webrtc_vad)
