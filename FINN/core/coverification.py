# Copyright (c) 2018, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import caffe
import numpy as np
import lmdb
import pickle
import pdb
import nn
import sys
from datasets import loadMNIST
from datasets import getImageNetVal_1ksubset_LMDB

# co-verify a FINN model against a Caffe model on a given dataset.

def verifyAgainstCaffe_ImageNet(pipeline, prototxt, caffemodel):
    testDB = "/home/maltanar/sandbox/hwgq/examples/imagenet/ilsvrc12_val_lmdb"
    labeltxtfile = "/home/maltanar/sandbox/hwgq/data/ilsvrc12/synset_words.txt"
    ok = 0
    nok = 0
    ok_caffe = 0
    nok_caffe = 0
    i = 0

    # Caffe Net for co-verification
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.blobs["data"].reshape(1, 3, 227, 227)

    # open the imagenet test images database
    lmdb_env = lmdb.open(testDB)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    # load label text for human-readable output
    labeltxt = np.loadtxt(labeltxtfile, str, delimiter='\t')

    def displayLabels(golden, predicted, prd_probs):
        top_k = prd_prob.argsort()[-1:-6:-1]
        print "Predicted: %d with %f probability" % (predicted, prd_prob[predicted])
        print labeltxt[top_k]
        print "Expected: %d" % (golden)
        print labeltxt[golden]

    for k, v in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(v)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = np.asarray(image.astype(np.float32))
        # subtract image mean
        image[0] = image[0] - 104
        image[1] = image[1] - 117
        image[2] = image[2] - 123
        # center crop to 227x227
        image = image[:, 15:15+227, 15:15+227]
        image = image.reshape((1, 3, 227, 227))
        # get result using Caffe
        caffe_out = net.forward_all(data=image)
        caffe_out_prob = caffe_out["prob"].flatten()
        #caffe_out_prob = net.blobs["fc8"].data.flatten()
        #plabel = np.argmax(caffe_out_prob)
        # get result using our Python flow
        (out, intm) = nn.execPipeline(image.flatten(), pipeline)
        plabel = np.argmax(out)
        if plabel == label:
            ok += 1
        else:
            nok += 1
        if np.isclose(caffe_out_prob, out, atol=1e-05).all():
            ok_caffe +=1
        else:
            nok_caffe += 1
            print("Difference: %f" % max(caffe_out_prob - out))
            # uncomment to track down differing cases in pdb
            #pdb.set_trace()
        i += 1

        #if i % 10 == 0:
        print("Progress: %d, ok = %d nok = %d" % (i,ok,nok))
        print("Progress: %d, ok_caffe = %d nok_caffe = %d" % (i,ok_caffe,nok_caffe))

def verifyAgainstCaffe_MNIST(pipeline, prototxt, caffemodel):
    # TODO separate and generalize the data loading
    (X_test, y_test) = loadMNIST()
    ok = 0
    nok = 0
    ok_caffe = 0
    nok_caffe = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.blobs["data"].reshape(1, 1, 28, 28)
    for i in range(10000):
        # TODO need to generalize input data preprocessing somehow
        #img = 2 * X_test[i][0] - 1
        img = X_test[i][0]
        img = np.asarray(img.flatten())
        (res, intm) = nn.execPipeline(img, pipeline)
        caffe_out = net.forward_all(data = img.reshape((1, 1, 28, 28)))
        if np.isclose(caffe_out.values()[0], res, atol=1e-05).all():
            ok_caffe +=1
        else:
            nok_caffe += 1
        res = np.argmax(res)
        if res == y_test[i]:
            ok += 1
        else:
            nok += 1
        if i % 10 == 0:
            print("Progress: %d, ok = %d nok = %d" % (i,ok,nok))
            print("Progress: %d, ok_caffe = %d nok_caffe = %d" % (i,ok_caffe,nok_caffe))

    print("Succeeded = %d failed = %d" % (ok, nok))
    print("Caffe output match = %d failed = %d" % (ok_caffe, nok_caffe))

def startProgress():
    # prepare to display progress on a separate line
    sys.stdout.write("\n")
    sys.stdout.flush()

def displayProgress(current_i, max_i, extra_info=""):
    # display progress
    sys.stdout.write("\rProgress: %d of %d %s" % (current_i, max_i, extra_info))
    sys.stdout.flush()

def endProgress():
    # print newline at the end of progress display
    sys.stdout.write("\n")
    sys.stdout.flush()

def testOnMNIST(net_to_test, num_images_to_test):
    "Test a given net on the MNIST test dataset."
    (X_test, y_test) = loadMNIST()
    ok = 0
    nok = 0
    i = 0
    startProgress()
    for i in range(num_images_to_test):
        img = X_test[i][0]
        img = np.asarray(img.flatten())
        (res, intm) = net_to_test.execPipeline(img)
        res = np.argmax(res)
        if res == y_test[i]:
            ok += 1
        else:
            nok += 1
        displayProgress(i+1, num_images_to_test, "(ok %d nok %d)" % (ok, nok))
    endProgress()
    return (ok, nok)

def testOnImageNet1kSubset(net_to_test, num_images_to_test):
    "Test a given net on 1k images from the ImageNet validation dataset."
    testDB = getImageNetVal_1ksubset_LMDB()
    ok = 0
    nok = 0
    i = 0
    # open the test images database
    lmdb_env = lmdb.open(testDB)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    startProgress()
    for k, v in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(v)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = np.asarray(image.astype(np.float32))
        # TODO data preprocessing should not be hardcoded
        # subtract image mean
        image[0] = image[0] - 104
        image[1] = image[1] - 117
        image[2] = image[2] - 123
        # center crop to 227x227
        image = image[:, 15:15+227, 15:15+227]
        image = image.reshape((1, 3, 227, 227))
        (res, intm) = net_to_test.execPipeline(image.flatten())
        plabel = np.argmax(res)
        if plabel == label:
            ok += 1
        else:
            nok += 1
        i += 1
        displayProgress(i, num_images_to_test, "(ok %d nok %d)" % (ok, nok))
        if i == num_images_to_test:
            break
    lmdb_env.close()
    endProgress()
    return (ok, nok)
