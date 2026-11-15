# This is source selection for an unqualified refresh experiment, not retirement
# of the two backports. Original iOS causal controls remain required (#20).
function(sd_native_apple_allocator_selection ggml_commit ggml_dir output)
    if(ggml_commit STREQUAL "a5915eb0e47adef21c3f25b889a008925ea838c2")
        set(${output} "backport" PARENT_SCOPE)
    elseif(ggml_commit STREQUAL "eeda9ec48fc84a3df0bb9dcc19ae9053075a2539")
        # Bind the reviewed Apple/TSAN and Metal shared+owned branches to exact
        # source bytes. Never silently skip a patch on a different revision.
        # Match the patch copier's LF normalization for Windows checkouts.
        file(READ "${ggml_dir}/src/ggml.c" cpu_text)
        file(READ "${ggml_dir}/src/ggml-metal/ggml-metal-device.m" metal_text)
        string(REPLACE "\r\n" "\n" cpu_text "${cpu_text}")
        string(REPLACE "\r\n" "\n" metal_text "${metal_text}")
        string(SHA256 cpu_sha "${cpu_text}")
        string(SHA256 metal_sha "${metal_text}")
        if(NOT cpu_sha STREQUAL "30ad3f6115025239236a2f6aa05aadf2358afac56b50ca9a4659640bcd63b860" OR
           NOT metal_sha STREQUAL "63755fe16b642f3346f8869d6602d076ac95c12bcb856f6817077be3281a9d51")
            message(FATAL_ERROR "Reviewed upstream Apple allocator sources changed; re-audit #20")
        endif()
        set(${output} "upstream-experimental" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unreviewed ggml allocator revision ${ggml_commit}; re-audit #20")
    endif()
endfunction()
