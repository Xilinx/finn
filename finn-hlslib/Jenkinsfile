/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

node {
    def app

    stage('Clone repository') {
        /* Let's make sure we have the repository cloned to our workspace */
        checkout scm
    }

    parallel firstBranch: {
        stage('Run tests SWG') {
              env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_swg.tcl')
    }
    }, secondBranch: {
        stage('Run tests POOL') {
              env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_pool.tcl')
    }
    }, thirdBranch: {
        stage('Run tests DWC') {
              env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_dwc.tcl')
    }
    }, fourthBranch: {
        stage('Run tests ADD') {
              env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_add.tcl')
    }
    }, fifthBranch: {
        stage('Run tests DUP_STREAM') {
              env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_dup_stream.tcl')
    }
    }, sixthBranch: {
        stage('Set-up virtual env') {
            env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('virtualenv venv; source venv/bin/activate;pip3.7 install -r requirements.txt')
        }
        stage('Generate weigths fro conv test') {
            sh('source venv/bin/activate; cd tb; python3.7 gen_weigths.py;')
        }
        stage('Run tests CONV3') {
            env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_conv3.tcl')
        }
        stage('Run tests CONVMMV') {
            env.FINN_HLS_ROOT = "${env.WORKSPACE}"
            echo "${env.FINN_HLS_ROOT}"
            sh('source /proj/xbuilds/2019.1_released/installs/lin64/Vivado/2019.1/settings64.sh; cd tb; vivado_hls -f test_convmmv.tcl')
        }
    }
}
