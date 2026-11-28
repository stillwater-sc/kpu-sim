# ============================================================================
# tools/csp-coverage/write_fails.cmake
# kpu-csp-coverage --write on a read-only copy of the report must exit nonzero and say it could not
# write it. Read-only does not stop root or bind Windows the same way, so there the test is skipped.
#   cmake -DTOOL=... -DDEPLOY_DIR=... -DDOC=... -DOUT=... -P write_fails.cmake
# ============================================================================
if(WIN32)
    message("SKIP: read-only files are not tested on Windows")
    return()
endif()
execute_process(COMMAND id -u OUTPUT_VARIABLE uid OUTPUT_STRIP_TRAILING_WHITESPACE)
if(uid STREQUAL "0")
    message("SKIP: root writes a read-only file")
    return()
endif()
if(EXISTS ${OUT})
    file(CHMOD ${OUT} PERMISSIONS OWNER_READ OWNER_WRITE)
endif()
file(READ ${DOC} text)
file(WRITE ${OUT} "${text}")
file(CHMOD ${OUT} PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
execute_process(COMMAND ${TOOL} --deploy-dir ${DEPLOY_DIR} --write ${OUT}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
file(CHMOD ${OUT} PERMISSIONS OWNER_READ OWNER_WRITE)
if(rc EQUAL 0 OR NOT err MATCHES "could not write")
    message(FATAL_ERROR "--write on a read-only report exited ${rc}:\n${out}${err}")
endif()
