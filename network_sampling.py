import argparse, logging, csv, datetime
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.lgbayesiannetwork import LGBayesianNetwork

# parse argumenents
argparser = argparse.ArgumentParser(prog='network_sampling',
                                    usage="this program randomly samples from a predefined bayes network",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--network", help="File path to network JSON file.")
argparser.add_argument("--output", help="Output path to save the csv file.",
                       default="untitled.csv")
argparser.add_argument("--size", help="sample size", type=int, default=50)
argparser.add_argument("--log_level", default='DEBUG',
                       help="Set the log level in console, you can set [DEBUG, ERROR, INFO, WARNING]")
args = argparser.parse_args()

input_network = args.network
output_csv = args.output
n_size = args.size

if str.lower(args.log_level) == 'debug':
   console_log_level = logging.DEBUG
elif str.lower(args.log_level) == 'info':
   console_log_level = logging.INFO
elif str.lower(args.log_level) == 'warning':
   console_log_level = logging.WARNING
elif str.lower(args.log_level) == 'error':
   console_log_level = logging.ERROR
else:
   print 'Log level is not set appropriately, see help by --help.'
   console_log_level = logging.DEBUG

logging.basicConfig(format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('network_sampling.log')
logger.setLevel(console_log_level)

# load nodedata and graphskeleton
nd = NodeData()
nd.load(input_network)
skel = GraphSkeleton()
skel.load(input_network)
skel.toporder()

# load bayesian network
lgbn = LGBayesianNetwork(skel, nd)

# sample
samples = lgbn.randomsample(n_size)

# write out csv
with open(output_csv, 'w') as csvfile:
   writer = csv.DictWriter(csvfile, fieldnames=samples[0].keys())
   writer.writeheader()
   cnt = 0
   write_begin = datetime.datetime.now()
   logger.debug(msg="write samples to file: {0:s}.".format(output_csv))
   for sample in samples:
      writer.writerow(sample)
      cnt += 1
      if cnt % 100 == 0:
         logger.debug(msg="write out {0:d}/{1:d} sample to csv file.".format(cnt, n_size))
   logger.debug(msg='write out done in {0:f} sec!'.format((datetime.datetime.now() - write_begin).total_seconds()))

