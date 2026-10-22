# Built as part of VCMI via its EXTRA_AI_DIRS option
add_library(AlphaBattleAI SHARED
	AlphaBattleAI.cpp Dump.cpp Encoder.cpp Model.cpp Policy.cpp Tensors.cpp main.cpp
	AlphaBattleAI.h Dump.h Encoder.h Model.h Policy.h Schema.generated.h StdInc.h Tensors.h
)
target_include_directories(AlphaBattleAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(AlphaBattleAI PRIVATE vcmi onnxruntime::onnxruntime)
vcmi_set_output_dir(AlphaBattleAI "AI")

# Exports the id -> name/stats tables the Python side uses (scripts/export_vocab.sh)
add_executable(alpha_export_vocab tools/ExportVocab.cpp)
target_include_directories(alpha_export_vocab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_export_vocab PRIVATE vcmi)
vcmi_set_output_dir(alpha_export_vocab "")
