# Small diagnostic utilities. Built unconditionally when the parent
# library is built standalone — these are nice to have on dev hosts
# but cost nothing to skip when CrispASR is consumed as a bound
# library.

if (CRISPASR_DFN)
    # dfn-denoise-wav: read a mono 16-bit WAV, push it through the
    # GGML DFN3 streaming pipeline, write a 48 kHz mono WAV.
    # Companion to /tmp/denoise_onnx_ref.py for A/B diagnostics
    # against the upstream ONNX-driven reference.
    add_executable(dfn-denoise-wav dfn-denoise-wav.cpp)
    target_include_directories(dfn-denoise-wav PRIVATE
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(dfn-denoise-wav PRIVATE crispasr)
endif()

