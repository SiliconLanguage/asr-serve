cmake_minimum_required(VERSION 3.22)
project(vad-edge C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Find DPDK ──────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# ── Find libbpf ───────────────────────────────────────────────────────────────
pkg_check_modules(LIBBPF REQUIRED libbpf>=1.3)

# ── Find wasmtime C API ───────────────────────────────────────────────────────
# Expects WASMTIME_PREFIX to be set, e.g. /usr/local.
# Download from: https://github.com/bytecodealliance/wasmtime/releases
if(NOT DEFINED WASMTIME_PREFIX)
    set(WASMTIME_PREFIX "/usr/local")
endif()
set(WASMTIME_INCLUDE_DIR "${WASMTIME_PREFIX}/include")
set(WASMTIME_LIB         "${WASMTIME_PREFIX}/lib/libwasmtime.a")

# ── BPF/XDP object (compiled by clang, not CMake) ────────────────────────────
# The XDP program must be compiled separately with:
#   clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
#         -I/usr/include/bpf -c src/xdp_vad.c -o xdp_vad.o
# A convenience custom target is provided for reference.
find_program(CLANG clang REQUIRED)
add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/xdp_vad.o
    COMMAND ${CLANG} -O2 -g -target bpf -D__TARGET_ARCH_x86
            -I/usr/include/bpf
            -c ${CMAKE_CURRENT_SOURCE_DIR}/src/xdp_vad.c
            -o ${CMAKE_CURRENT_BINARY_DIR}/xdp_vad.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/xdp_vad.c
    COMMENT "Compiling eBPF program xdp_vad.o"
)
add_custom_target(xdp_object ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/xdp_vad.o)

# ── vad_loader binary ─────────────────────────────────────────────────────────
add_executable(vad_loader src/vad_loader.c)
target_include_directories(vad_loader PRIVATE ${LIBBPF_INCLUDE_DIRS})
target_link_libraries(vad_loader PRIVATE ${LIBBPF_LIBRARIES} elf z)
target_compile_options(vad_loader PRIVATE ${LIBBPF_CFLAGS_OTHER} -Wall -Wextra)
add_dependencies(vad_loader xdp_object)

# ── dpdk_pipeline binary ──────────────────────────────────────────────────────
add_executable(dpdk_pipeline src/dpdk_pipeline.c)
target_include_directories(dpdk_pipeline PRIVATE
    ${DPDK_INCLUDE_DIRS}
    ${WASMTIME_INCLUDE_DIR}
)
target_link_libraries(dpdk_pipeline PRIVATE
    ${DPDK_LIBRARIES}
    ${WASMTIME_LIB}
    pthread dl m
)
target_compile_options(dpdk_pipeline PRIVATE
    ${DPDK_CFLAGS_OTHER}
    -march=native
    -Wall -Wextra
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS vad_loader dpdk_pipeline RUNTIME DESTINATION bin)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xdp_vad.o DESTINATION lib/bpf)
install(FILES wasm/vad_module.wat DESTINATION share/vad-edge)
