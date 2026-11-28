# ============================================================================
# tools/csp-check/gen_then_check.cmake
# kpu-csp-gen writes a distributed gather to OUT, then kpu-csp-check checks it (D0-D4) and runs it
# (D5) on DEPLOY. Fails unless both exit 0 and the check reports D5 passing.
#   cmake -DGEN=... -DCHECK=... -DDEPLOY=... -DOUT=... -P gen_then_check.cmake
# ============================================================================
execute_process(COMMAND ${GEN} --algo gather --size 32 --tile 8 --target ${DEPLOY} --distributed -o ${OUT}
                RESULT_VARIABLE gen_rc OUTPUT_VARIABLE gen_out ERROR_VARIABLE gen_err)
if(NOT gen_rc EQUAL 0)
    message(FATAL_ERROR "kpu-csp-gen exited ${gen_rc}:\n${gen_out}${gen_err}")
endif()
execute_process(COMMAND ${CHECK} --distributed ${OUT} --deploy ${DEPLOY} --run
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0 OR NOT "${out}${err}" MATCHES "D5.pass")
    message(FATAL_ERROR "kpu-csp-check exited ${rc}:\n${out}${err}")
endif()
