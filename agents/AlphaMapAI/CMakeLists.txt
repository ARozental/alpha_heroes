# Built as part of VCMI via its EXTRA_AI_DIRS option. The adventure-map AI of campaign_model/: every decision is the
# model's, asked of the inference server (battle_model/inference.py, family "campaign"); battles are a battle AI's.
set(ALPHA_MAP_AI_SOURCES
	AlphaMapAI.cpp Actions.cpp Encoder.cpp Ledger.cpp Tensors.cpp main.cpp ../AlphaBattleAI/InferenceClient.cpp
	AlphaMapAI.h Ledger.h Schema.generated.h StdInc.h Tensors.h ../AlphaBattleAI/InferenceClient.h
)
find_package(ZLIB REQUIRED)
add_library(AlphaMapAI SHARED ${ALPHA_MAP_AI_SOURCES})
target_include_directories(AlphaMapAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../AlphaBattleAI ${CMAKE_SOURCE_DIR})
target_link_libraries(AlphaMapAI PRIVATE vcmi ZLIB::ZLIB)
vcmi_set_output_dir(AlphaMapAI "AI")

# Writes a random map (VCMI's generator) for self-play games (campaign_model/play.py)
add_executable(alpha_make_random_map tools/MakeRandomMap.cpp)
target_include_directories(alpha_make_random_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_make_random_map PRIVATE vcmi)
vcmi_set_output_dir(alpha_make_random_map "")

add_executable(alpha_map_info tools/MapInfo.cpp)
target_include_directories(alpha_map_info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_map_info PRIVATE vcmi)
vcmi_set_output_dir(alpha_map_info "")
