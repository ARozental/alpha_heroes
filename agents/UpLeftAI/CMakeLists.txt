# Built as part of VCMI via its EXTRA_AI_DIRS option
add_library(UpLeftAI SHARED UpLeftAI.cpp main.cpp StdInc.h UpLeftAI.h)
target_include_directories(UpLeftAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(UpLeftAI PRIVATE vcmi)
vcmi_set_output_dir(UpLeftAI "AI")
