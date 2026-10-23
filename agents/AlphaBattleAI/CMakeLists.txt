# Built as part of VCMI via its EXTRA_AI_DIRS option
set(ALPHA_BATTLE_AI_SOURCES
	AlphaBattleAI.cpp BattleSim.cpp Dump.cpp Encoder.cpp Model.cpp Policy.cpp Tensors.cpp main.cpp
	AlphaBattleAI.h BattleSim.h Dump.h Encoder.h Model.h Policy.h Schema.generated.h StdInc.h Tensors.h
)
find_package(onnxruntime CONFIG REQUIRED)
# AlphaBattleAIDev: the same AI under another name, to try C++ changes while training runs keep using
# AlphaBattleAI: `cmake --build build --target AlphaBattleAIDev`, then play with ai="AlphaBattleAIDev"
foreach(target AlphaBattleAI AlphaBattleAIDev)
	add_library(${target} SHARED ${ALPHA_BATTLE_AI_SOURCES})
	target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
	# vcmiservercommon: the server's battle rules, run on copies of a battle by BattleSim
	target_link_libraries(${target} PRIVATE vcmi vcmiservercommon onnxruntime::onnxruntime)
	vcmi_set_output_dir(${target} "AI")
endforeach()

# Exports the id -> name/stats tables the Python side uses (scripts/export_vocab.sh)
add_executable(alpha_export_vocab tools/ExportVocab.cpp)
target_include_directories(alpha_export_vocab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_export_vocab PRIVATE vcmi)
vcmi_set_output_dir(alpha_export_vocab "")

# Builds a battle-only map from a battle_model scenario file (battle_model/play.py)
add_executable(alpha_make_battle_map tools/MakeBattleMap.cpp)
target_include_directories(alpha_make_battle_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_make_battle_map PRIVATE vcmi)
vcmi_set_output_dir(alpha_make_battle_map "")
