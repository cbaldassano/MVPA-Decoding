This software performs cross-validated linear SVM classification, intended
for use in MVPA decoding of fMRI data. The included files are:

-validateAndTest.m: Main function, performs ROI decoding
-decodeSearchlight.m: Wrapper around validateAndTest for performing
    searchlight-style decoding of multiple spatial spheres
-example.m: Demo program illustrating the use of these functions
[svmtrain/svmtest_wrapper: internal functions for classification]

This software requires the LIBSVM Matlab library, available from
    http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Developed on Matlab 7.13.0.564 (R2011b) and LIBSVM 3.11, but should run in
    most versions. For very old versions of Matlab, tilde (~) characters
    will need to be manually replaced with dummy variables (any unused name)



Copyright (c) 2015, Christopher Baldassano, Stanford University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Stanford University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER BALDASSANO BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
