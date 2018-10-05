

add_library(Solver SHARED species.h common.h arrayutils.h relativisticspecies.h species.cpp arrayutils.cpp relativisticspecies.cpp)
TARGET_LINK_LIBRARIES(Solver ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})