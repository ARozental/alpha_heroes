# Built as part of VCMI via its EXTRA_AI_DIRS option
add_library(AlphaBattleAI SHARED
	AlphaBattleAI.cpp Dump.cpp Encoder.cpp Model.cpp Policy.cpp Tensors.cpp main.cpp
	AlphaBattleAI.h Dump.h Encoder.h Model.h Policy.h Schema.generated.h StdInc.h Tensors.h
)
target_include_directories(AlphaBattleAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(AlphaBattleAI PRIVATE vcmi onnxruntime::onnxruntime)
vcmi_set_output_dir(AlphaBattleAI "AI")
