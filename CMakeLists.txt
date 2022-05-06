cmake_minimum_required(VERSION 3.15)
project(SecureKVM)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread")

set(MALICIOUS_LIB malicious_lib/reed_solomn_reconstruct.h malicious_lib/reed_solomn_reconstruct.cpp malicious_lib/feldman_vss.cpp malicious_lib/feldman_vss.h)
set(SOURCE_FILES_BASE Constant.cpp Constant.h util/SocketOnline.cpp util/SocketOnline.h util/SocketManager.cpp util/SocketManager.h Mat.cpp Mat.h Player.cpp Player.h Op.cpp Op.h NeuronMat.cpp NeuronMat.h machine_learning/NN.cpp machine_learning/NN.h MathOp.cpp MathOp.h util/IOManager.cpp util/IOManager.h decision_tree/NodeMat.cpp decision_tree/NodeMat.h)
set(SOURCE_FILES_OMEN comen/errorHandler.c comen/errorHandler.h comen/common.h comen/common.c comen/smoothing.c comen/smoothing.h comen/commonStructs.c comen/commonStructs.h comen/nGramReader.c comen/nGramReader.h)
set(RNADOMNESS_LIB util/Randomness.cpp util/Randomness.h)
set(PEMUTATION_LIB util/Permutation.cpp util/Permutation.h)

#add_executable(reed malicious_lib/reed_solomn_reconstruct.cpp malicious_lib/reed_solomn_reconstruct.h Constant.h Constant.cpp malicious_lib/feldman_vss.h malicious_lib/feldman_vss.cpp)
#add_executable(feldman malicious_lib/feldman_vss.cpp malicious_lib/feldman_vss.h)

# DT Inference
add_executable(DT_MPC decision_tree/DTDAG.cpp decision_tree/DTDAG.h decision_tree/DT_graph.cpp decision_tree/DT_graph.h decision_tree/DT_main.cpp decision_tree/DT_main.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} decision_tree/NodeMat.cpp decision_tree/NodeMat.h ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
# Markov Inference
add_executable(evalPW_mpc mpc_omen/omen_eval.cpp mpc_omen/omen_eval.h comen/evalPW.c comen/evalPW.h comen/cmdlineEvalPW.c comen/cmdlineEvalPW.h ${SOURCE_FILES_OMEN} ${SOURCE_FILES_BASE} mpc_omen/NgramGraph.cpp mpc_omen/NgramGraph.h ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})


# Benchmarks of Primitives
add_executable(test_factory benchmark/TestFactory.cc benchmark/TestFactory.h benchmark/GetBenchmark.cc benchmark/GetBenchmark.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
# add_executable(test_opappro test_op/TestMathOp.cpp test_op/TestMathOp.h  kv_statics/StatisticsGraph.cpp kv_statics/StatisticsGraph.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
# add_executable(test_perm test_op/TestMathOp.cpp test_op/TestMathOp.h  kv_statics/StatisticsGraph.cpp kv_statics/StatisticsGraph.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
# add_executable(test_reshare test_op/TestMathOp.cpp test_op/TestMathOp.h  kv_statics/StatisticsGraph.cpp kv_statics/StatisticsGraph.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(genperm_func test_op/test_genperm.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(shuffledata_func test_op/test_shuffledata.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(secgetper_func test_op/test_secgetper.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(log_func test_op/test_log.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(test_secgetper test_op/secgetper_time.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})
add_executable(test_log test_op/log_time.cpp test_op/TestMathOp.cpp test_op/TestMathOp.h ${SOURCE_FILES_BASE} ${MALICIOUS_LIB} ${RNADOMNESS_LIB} ${PEMUTATION_LIB})


#target_link_libraries(omen ws2_32)
#target_link_libraries(DT_MPC ws2_32)