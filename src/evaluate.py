# MapTD
# Copyright (C) 2018 Jerod Weinman, Nathan Gifford, Abyaya Lamsal, 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import data_tools
import stats
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--gt_path", type=str, default='../data/test',
                    help='Base directory for ground truth data')
parser.add_argument("--pred_path", type=str, default='../data/predict',
                    help='Base directory for predicted output data')
parser.add_argument("--score_thresh", type=float, default=None,
                    help='Score threshold to filter by')
parser.add_argument("--filename_pattern", type=str, default='*',
                    help='File pattern for data')
parser.add_argument("--iou_thresh", type=float, default=0.5,
                    help='Intersection-over-Union threshold for match')
parser.add_argument("--match_labels", action="store_true", default=False,
                    help='Whether to require labels to match')
parser.add_argument("--json", action="store_true", default=False,
                    help='Ground truth json or txt, default txt')
parser.add_argument("--save_result", type=str, default=None,
                    help='JSON file in which to save results in pred_path')

def threshold_predictions(polys,labels,scores, args):

  t_polys = list()
  t_labels = list()
  t_scores = list()

  for (poly,label,score) in zip(polys,labels,scores):
    if score > args.score_thresh:
      t_polys.append(poly)
      t_labels.append(label)
      t_scores.append(score)
  return t_polys,t_labels,t_scores


def main(args):
    """Loads up ground truth and prediction files, calculates and
       prints statistics
    """

    # Load file lists
    prediction_files = data_tools.get_filenames(
      args.pred_path,
      str.split(args.filename_pattern,','),
      'txt')
    
    if args.json:
        ext_paired = 'json'
    else:
        ext_paired = 'txt'
    
    ground_truth_files = data_tools.get_paired_filenames(
      prediction_files, args.gt_path, ext_paired)

    assert len(ground_truth_files) == len(prediction_files)

    # Load files contents and package for stats evaluation
    predictions = {}
    ground_truths = {}
    
    for pred_file,truth_file in zip(prediction_files,ground_truth_files):

      base = os.path.splitext(os.path.basename(pred_file))[0]
      
      if args.json:
        [_,gt_polys,gt_labels,_] = data_tools.parse_boxes_from_text( truth_file )
      else:
        [_,gt_polys,gt_labels] = data_tools.parse_boxes_from_json( truth_file )
      [_,polys,labels,scores] = data_tools.parse_boxes_from_text( pred_file )

      if args.score_thresh: # Filter predictions if necessary
        polys,labels,scores = threshold_predictions(
          polys, labels, scores, args)

      predictions[base] =  { 'polygons' : polys,
                             'labels'   : labels,
                             'scores'   : scores }
      ground_truths[base] = {'polygons' : gt_polys,
                             'labels'   : gt_labels }

    # Calculate statistics on predictions for ground truths
    sample_stats,total_stats = stats.evaluate_predictions(
      ground_truths,
      predictions,
      match_labels=args.match_labels,
      iou_match_thresh=args.iou_thresh)

    # Display save the results
    print(sample_stats)
    print(total_stats)
    
    if args.save_result:
      import json
      with open(os.path.join(args.pred_path,args.save_result+'.json'),'w') \
           as fd:
        json.dump({'individual': sample_stats, 'overall': total_stats}, fd,
                  indent=4)

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
     
