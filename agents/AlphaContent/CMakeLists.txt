# The content match (docs/hota_transfer_plan.md, Part 1): a static library both agents, the exporter and the map tools
# link. Added by AlphaBattleAI's CMakeLists (no machine's EXTRA_AI_DIRS changes)
add_library(alpha_content STATIC Content.cpp Content.h StdInc.h)
target_include_directories(alpha_content PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_content PUBLIC vcmi)
set_target_properties(alpha_content PROPERTIES POSITION_INDEPENDENT_CODE ON) # (linked into the agents' shared libraries)

# The match from the command line: the engine (VCMI_USER_DIR's mods) against a vocabulary
add_executable(alpha_content_check tools/ContentCheck.cpp)
target_include_directories(alpha_content_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(alpha_content_check PRIVATE alpha_content vcmi)
vcmi_set_output_dir(alpha_content_check "")
