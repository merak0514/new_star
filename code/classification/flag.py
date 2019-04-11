import sys, os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('training_dataset',None,'training-dataset dir')
flags.DEFINE_string('model','./model25/','checkpoint save path')
flags.DEFINE_string('prediction_file','./submit.csv','submit.scv save path')
flags.DEFINE_string('gpu', '0', 'comma separated list of GPU to use.')
 
def main(argv):
    del argv
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    print(FLAGS.training_dataset,FLAGS.model,FLAGS.prediction_file,FLAGS.gpu)
    
 
if __name__ == '__main__':
    app.run(main)