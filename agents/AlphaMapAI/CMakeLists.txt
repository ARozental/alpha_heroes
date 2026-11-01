# Built as part of VCMI via its EXTRA_AI_DIRS option. The adventure-map AI of campaign_model/: every decision is the
# model's, asked of the inference server (battle_model/inference.py, family "campaign"); battles are a battle AI's.
set(ALPHA_MAP_AI_SOURCES
	AlphaMapAI.cpp Actions.cpp Advice.cpp Encoder.cpp Ledger.cpp Match.cpp Tensors.cpp main.cpp ../AlphaBattleAI/InferenceClient.cpp
	AlphaMapAI.h Ledger.h Schema.generated.h StdInc.h Tensors.h ../AlphaBattleAI/InferenceClient.h
)
find_package(ZLIB REQUIRED)
add_library(AlphaMapAI SHARED ${ALPHA_MAP_AI_SOURCES})
target_include_directories(AlphaMapAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../AlphaBattleAI ${CMAKE_SOURCE_DIR})
target_link_libraries(AlphaMapAI PRIVATE vcmi ZLIB::ZLIB)
vcmi_set_output_dir(AlphaMapAI "AI")
# Nullkiller2's advice (Advice.cpp, ALPHA_MAP_NK2_ADVICE): a Nullkiller2 of each player's, asked what it would do
if(TARGET Nullkiller2)
	target_include_directories(AlphaMapAI PRIVATE ${CMAKE_SOURCE_DIR}/AI/Nullkiller2)
	target_link_libraries(AlphaMapAI PRIVATE Nullkiller2)
	target_compile_definitions(AlphaMapAI PRIVATE ALPHA_MAP_NK2_ADVICE)
endif()

# Nullkiller2 as the campaign model's teacher (Teacher.cpp): Nullkiller2 playing, each of its turn requests recorded as
# our decision doing the same (VCMI_ADVENTURE_AI_<COLOR>=AlphaTeacherAI; campaign_model/teacher.py)
if(TARGET Nullkiller2)
	list(REMOVE_ITEM ALPHA_MAP_AI_SOURCES main.cpp)
	add_library(AlphaTeacherAI SHARED ${ALPHA_MAP_AI_SOURCES} Teacher.cpp)
	target_include_directories(AlphaTeacherAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../AlphaBattleAI ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/AI/Nullkiller2)
	target_link_libraries(AlphaTeacherAI PRIVATE vcmi ZLIB::ZLIB Nullkiller2)
	target_compile_definitions(AlphaTeacherAI PRIVATE ALPHA_MAP_NK2_ADVICE)
	vcmi_set_output_dir(AlphaTeacherAI "AI")
endif()

# Writes a random map (VCMI's generator) for self-play games (campaign_model/play.py)
add_executable(alpha_make_random_map tools/MakeRandomMap.cpp)
target_include_directories(alpha_make_random_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_make_random_map PRIVATE vcmi)
vcmi_set_output_dir(alpha_make_random_map "")

add_executable(alpha_map_info tools/MapInfo.cpp)
target_include_directories(alpha_map_info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_map_info PRIVATE vcmi)
vcmi_set_output_dir(alpha_map_info "")
