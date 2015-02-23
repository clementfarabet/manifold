CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

SET(src generic/sptree.cpp generic/tsne.cpp)
SET(luasrc init.lua laplacian_eigenmaps.lua lle.lua tsne.lua)

ADD_TORCH_PACKAGE(manifold "${src}" "${luasrc}" "Manifold")
TARGET_LINK_LIBRARIES(manifold luaT TH)
INSTALL(FILES "README.md" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/manifold")