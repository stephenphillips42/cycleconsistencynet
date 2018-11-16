import os
import sys
import glob
import numpy as np
import argparse
import argcomplete
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import tqdm
# import yaml

import myutils
import options

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_latex_opts():
  parser = argparse.ArgumentParser(description='Experiment with output')
  argcomplete.autocomplete(parser)
  parser.add_argument('--verbose',
                      default=False,
                      type=str2bool,
                      help='Print everything or not')
  parser.add_argument('--just_view',
                      default=False,
                      type=str2bool,
                      help='Just view, do not save')
  view_style_choices = [ 'default', 'paper' ]
  parser.add_argument('--view_style',
                      default=view_style_choices[0],
                      choices=view_style_choices,
                      help='Options for the figure style')
  parser.add_argument('--index',
                      default=0,
                      type=int,
                      help='Test data index to experiment with')
  parser.add_argument('--experiments',
                      nargs='+',
                      help='Path to test data to run analysis on')
  parser.add_argument('--plot_generate',
                      default='',
                      help='Path to experiment to generate single plot')
  parser.add_argument('--latex_path',
                      default='./journal',
                      help='Path to latex - very very machine dependent')
  parser.add_argument('--save_dir',
                      default='figures',
                      help='Path to latex - very very machine dependent')
  parser.add_argument('--viewer_size',
                      default=4,
                      type=int,
                      help='Run in debug mode')

  opts = parser.parse_args()
  # Finished, return options
  return opts

def npload(fdir,idx):
  return dict(np.load("{}/np_test-{:04d}.npz".format(fdir,idx)))

def get_sorted(labels):
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  return slabels, sorted_idxs

class Experiment(object):
  def __init__(self, folder, extention='png', verbose=False):
    if folder[-1] == '/':
      folder = folder[:-1]
    self.folder = folder
    self.name = os.path.split(folder)[1]
    self.extention = extention
    self.verbose = verbose
    self.view_style = opts.view_style

  def get_losses(self, use_time=False):
    loss = np.load(os.path.join(self.folder, 'loss.npy'))
    test_loss = np.load(os.path.join(self.folder, 'test_loss.npy'))
    if use_time:
      train_time = np.linspace(0,len(loss),len(loss))
      test_time = np.linspace(0,len(loss),len(test_loss))
      return loss, test_loss, train_time, test_time
    else:
      return loss, test_loss

  def generate_plot(self, size, index=0):
    if self.view_style == 'default':
      self.generate_plot_default(size, index=index)
    elif self.view_style == 'paper':
      self.generate_plot_paper(size, index=index)

  def generate_plot_default(self, size, index=0):
    nrows, ncols = 1, 5
    fig = plt.figure(figsize=(ncols*size,1.4*nrows*size))
    gs = GridSpec(4, 14)
    ax = [
      plt.subplot(gs[:,:4]),
      plt.subplot(gs[:,4:6]),
      plt.subplot(gs[1:,6:8]),
      plt.subplot(gs[:,8:11]),
      plt.subplot(gs[:,11:14]),
      plt.subplot(gs[0,6:8]),
    ]
    fig.tight_layout()

    ##### Plot losses
    loss, test_loss, train_time, test_time = self.get_losses(True)
    ax[0].plot(train_time, loss, test_time, test_loss)
    # print((np.mean(loss), np.std(loss), np.max(loss), np.min(loss)))
    ax[0].set_ylim([0, np.mean(loss)+4*np.std(loss)])
    ax[0].legend(['Train', 'Test'])
    ##### Plot a specific example
    ### Get specific example
    ldnames = sorted(glob.glob(os.path.join(self.folder, 'test*npz')))
    ld = np.load(ldnames[-1])
    emb_init = ld['input'][index]
    emb_gt = ld['gt'][index]
    emb_out = ld['output'][index]
    adjmat = ld['adjmat'][index]
    ### Create labels
    slabels, sorted_idxs = get_sorted(emb_gt)
    soutput = emb_out[sorted_idxs]
    srand = myutils.dim_normalize(emb_init[sorted_idxs])
    lsim = np.abs(np.dot(slabels, slabels.T))
    osim = np.abs(np.dot(soutput, soutput.T))
    rsim = np.abs(np.dot(srand, srand.T))
    ### Create plots
    if soutput.shape[1] == slabels.shape[1]:
      u, s, v = np.linalg.svd(np.dot(soutput.T, slabels))
      o_ = np.ones_like(s)
      o_[-1] = np.linalg.det(np.dot(u,v))
      Q = np.dot(u, np.dot(np.diag(o_), v))
      im0 = ax[1].imshow(np.abs(np.dot(soutput, Q)))
    else:
      im0 = ax[1].imshow(np.abs(soutput))
    im1 = ax[2].imshow(osim)
    # fig.colorbar(im0, ax=ax[1])
    fig.colorbar(im1, ax=ax[2])
    ### Plot Histogram
    diag = np.reshape(osim[lsim==1],-1)
    off_diag = np.reshape(osim[lsim==0],-1)
    baseline_diag = np.reshape(rsim[lsim==1],-1)
    baseline_off_diag = np.reshape(rsim[lsim==0],-1)
    ax[3].hist([ diag, baseline_diag ], bins=20, density=True,
             label=[ 'diag', 'baseline_diag' ])
    ax[3].legend()
    ax[3].set_title('Diagonal Similarity Rate')
    ax[4].hist([ off_diag, baseline_off_diag ], bins=20, density=True,
             label=[ 'off_diag', 'baseline_off_diag' ])
    ax[4].set_title('Off Diagonal Similarity Rate')
    ax[4].legend()
    ### Titles
    ax[0].set_title('Train/test loss')
    ax[1].set_title('Normalized Embeddings')
    ax[-1].set_title('Pairwise similarities')
    ax[-1].set_axis_off()
    
    # Show everything
    # plt.show()
    # break
    return fig

  def generate_plot_paper(self, size, index=0):
    nrows, ncols = 1, 4
    fig = plt.figure(figsize=(ncols*size,1.4*nrows*size))
    gs = GridSpec(4, 10)
    ax = [
      plt.subplot(gs[:,0:2]),
      plt.subplot(gs[1:,2:4]),
      plt.subplot(gs[:,4:7]),
      plt.subplot(gs[:,7:10]),
      plt.subplot(gs[0,2:4]),
    ]
    fig.tight_layout()

    ##### Plot a specific example
    ### Get specific example
    ldnames = sorted(glob.glob(os.path.join(self.folder, 'test*npz')))
    ld = np.load(ldnames[-1])
    emb_init = ld['input'][index]
    emb_gt = ld['gt'][index]
    emb_out = ld['output'][index]
    adjmat = ld['adjmat'][index]
    ### Create labels
    slabels, sorted_idxs = get_sorted(emb_gt)
    soutput = emb_out[sorted_idxs]
    srand = myutils.dim_normalize(emb_init[sorted_idxs])
    lsim = np.abs(np.dot(slabels, slabels.T))
    osim = np.abs(np.dot(soutput, soutput.T))
    rsim = np.abs(np.dot(srand, srand.T))
    ### Create plots
    if soutput.shape[1] == slabels.shape[1]:
      u, s, v = np.linalg.svd(np.dot(soutput.T, slabels))
      o_ = np.ones_like(s)
      o_[-1] = np.linalg.det(np.dot(u,v))
      Q = np.dot(u, np.dot(np.diag(o_), v))
      im0 = ax[0].imshow(np.abs(np.dot(soutput, Q)))
    else:
      im0 = ax[0].imshow(np.abs(soutput))
    im1 = ax[1].imshow(osim)
    # fig.colorbar(im0, ax=ax[1])
    fig.colorbar(im1, ax=ax[1])
    ### Plot Histogram
    diag = np.reshape(osim[lsim==1],-1)
    off_diag = np.reshape(osim[lsim==0],-1)
    baseline_diag = np.reshape(rsim[lsim==1],-1)
    baseline_off_diag = np.reshape(rsim[lsim==0],-1)
    ax[2].hist([ diag, baseline_diag ], bins=20, density=True,
             label=[ 'Same Point Output Similarity', 'Same Point Sift Similarity' ])
    ax[2].legend()
    ax[2].set_title('Diagonal Similarity Rate')
    ax[3].hist([ off_diag, baseline_off_diag ], bins=20, density=True,
             label=[ 'Diff. Point Output Similarity', 'Diff. Point Sift Similarity' ])
    ax[3].set_title('Off Diagonal Similarity Rate')
    ax[3].legend()
    ### Titles
    ax[0].set_title('Normalized Embeddings')
    ax[-1].set_title('Pairwise similarities')
    ax[-1].set_axis_off()
    
    # Show everything
    # plt.show()
    # break
    return fig

  def get_latex_name(self):
    """Simple name for saving plots."""
    return self.name[5:].replace('_','-') + '-'

  def get_plot_name(self):
    """Simple name for saving plots."""
    return self.name[5:].replace('_','-') + '.{}'.format(self.extention)

  def get_arch_type(self, arch_):
    arch_type = arch_
    if 'vanilla' in arch_:
      arch_type = 'Vanilla Type {}'.format(arch_[-1])
    elif 'longskip' in arch_:
      arch_type = 'Long Skip Type {}'.format(arch_[-1])
    elif 'skip' in arch_ :
      if arch_[-1] in [ '0', '1', '2' ]:
        arch_type = 'Skip Connections Type {}'.format(1+int(arch_[-1]))
      else:
        arch_type = 'Skip Connections Type 0'
    elif 'spattn' in arch_ :
      arch_type = 'Sparse GAT Network Type {}'.format(arch_[-1])
    elif 'attn' in arch_:
      arch_type = 'GAT Network Type {}'.format(arch_[-1])
    return arch_type

  # TODO: Make this nicer?
  def get_name(self):
    """Easily readble name for plot titles."""
    tokens = self.name.split('-')
    if tokens[0] != 'save':
      print("ERROR: Name not in right format: {}".format(fname), file=sys.stderr)
      sys.exit(1)
    name = ""
    idx = 1
    if tokens[idx] == 'synth':
      idx += 1
      name += 'Data type \\texttt{{synth{}}}'.format(tokens[idx])
    else:
      name += 'Data type \\texttt{{{}}}'.format(tokens[idx])
    idx += 1
    while idx < len(tokens):
      if tokens[idx] in options.arch_choices:
        name += ', {} Architecture'.format(self.get_arch_type(tokens[idx]))
      elif tokens[idx] in [ 'load', 'ld' ]:
        if idx+1 < len(tokens) and not str2bool(tokens[idx+1]):
          name += ', Data generation'
          idx += 1
      elif tokens[idx] in options.activation_types:
        name += ', Activ. {}'.format(tokens[idx].title())
      elif tokens[idx] in [ 'loss' ]:
        if idx + 1 >= len(tokens):
          print("ERROR: Name not in right format: {}".format(fname), file=sys.stderr)
          sys.exit(1)
        name += ', Loss {}'.format(tokens[idx+1].title())
        idx += 1
      elif tokens[idx] == 'unsup':
        if idx + 1 >= len(tokens):
          print("ERROR: Name not in right format: {}".format(fname), file=sys.stderr)
          sys.exit(1)
        if str2bool(tokens[idx+1]):
          name += ', Unsupervised'
        elif not str2bool(tokens[idx+1]):
          name += ', Supervised'
      elif tokens[idx] in [ 'type', 'lrdecaytype' ]:
        if idx + 1 >= len(tokens):
          print("ERROR: Name not in right format: {}".format(fname), file=sys.stderr)
          sys.exit(1)
        name += ', LR Decay {}'.format(tokens[idx + 1].title())
      elif tokens[idx] in [ 'steps', 'lrdecaysteps' ]:
        if idx + 1 >= len(tokens):
          print("ERROR: Name not in right format: {}".format(fname), file=sys.stderr)
          sys.exit(1)
        name += ', LR Decay Steps {}'.format(tokens[idx + 1].title())
      idx += 1
    return name

  def get_stats_all(self):
    ldnames = sorted(glob.glob(os.path.join(self.folder, 'test*npz')))
    ld = np.load(ldnames[-1])
    emb_init = ld['input']
    emb_gt = ld['gt']
    emb_out = ld['output']
    adjmat = ld['adjmat']
    n = len(emb_gt)
    stats = []
    for i in tqdm.tqdm(range(n), disable=not self.verbose):
      stats.append(self.get_sample_stats(emb_init[i], emb_gt[i], emb_out[i]))
    return np.mean(np.array(stats), 0)

  def get_sample_stats(self, emb_init, emb_gt, emb_out):
    slabels, sorted_idxs = get_sorted(emb_gt)
    soutput = emb_out[sorted_idxs]
    srand = myutils.dim_normalize(emb_init[sorted_idxs])
    lsim = np.abs(np.dot(slabels, slabels.T))
    osim = np.abs(np.dot(soutput, soutput.T))
    rsim = np.abs(np.dot(srand, srand.T))
    diag = np.reshape(osim[lsim==1],-1)
    off_diag = np.reshape(osim[lsim==0],-1)
    baseline_diag = np.reshape(rsim[lsim==1],-1)
    baseline_off_diag = np.reshape(rsim[lsim==0],-1)
    return (np.mean(diag), np.std(diag), \
            np.mean(off_diag), np.std(off_diag), \
            np.mean(baseline_diag), np.std(baseline_diag), \
            np.mean(baseline_off_diag), np.std(baseline_off_diag))

def build_latex_example():
  # Run experiment
  n = len(emb_gt)
  meanstats = np.abs(np.random.randn(12))
  prefix = "pairwise_"
  latex_string = """\\begin{{figure}}[H]
  \\begin{{minipage}}{{.45\\linewidth}}
      \\centering
      \\subfloat[Typical embedding and similarity matrix of pairwise noise]{{ \\includegraphics[width=0.9\\textwidth]{{figures/{0}output.png}} \\label{{fig:{0}sub1}} }}
  \\end{{minipage}}
  \\begin{{minipage}}{{.45\\linewidth}}
      \\centering
      \\subfloat[Typical histogram for similarities of pairwise noise]{{ \\includegraphics[width=0.9\\textwidth]{{figures/{1}hist.png}} \\label{{fig:{1}sub2}} }}
  \\end{{minipage}}
  \\centering
  \\subfloat[Numerical results]{{
      \\begin{{tabular}}{{|c|c|c|c|}} \\hline
                                     &    Trained Model     &    Baseline          \\\\ \\hline
          Same Point Similarity      & {1:.2e} $\\pm$ {2:.2e} & {3:.2e} $\\pm$ {4:.2e} \\\\ \\hline
          Distinct Point Similarity  & {5:.2e} $\\pm$ {6:.2e} & {7:.2e} $\\pm$ {8:.2e} \\\\ \\hline
      \\end{{tabular}}
      \\label{{fig:{0}_tab1}}
  }}
  \\label{{fig:pairwise3_large_plot}}
  \\end{{figure}}""".format(prefix,
                            meanstats[0],meanstats[1],
                            meanstats[2],meanstats[3],
                            meanstats[4],meanstats[5],
                            meanstats[6],meanstats[7])
  print(latex_string)

#     \\subfloat[Typical embedding and similarity matrix of pairwise noise]{{ \\includegraphics[width=0.9\\textwidth]{{figures/{1}output.png}} \\label{{fig:{1}sub1}} }}

class LatexGenerator(object):
  def __init__(self, opts, verbose=True):
    self.top_dir = opts.latex_path
    self.save_dir = opts.save_dir
    self.experiments = []
    self.index = opts.index
    self.viewer_size = opts.viewer_size
    self.verbose = opts.verbose
    self.just_view = opts.just_view
    for exp_ in tqdm.tqdm(opts.experiments, disable=not self.verbose):
      self.experiments.append(Experiment(exp_, verbose=opts.verbose))

  def save_images_and_output_latex(self, exp, caption="Typical sample", index=0):
    latex_string = "\\begin{{figure}}[H]\n"
    latex_string += "  \\includegraphics[width=0.9\\textwidth]{{{0}}}\n"
    latex_string += "  \\label{{fig:{1}sub1}}\n"
    latex_string += "  \\caption{{{2}}}\n"
    latex_string += "\\end{{figure}}\n"
    fname = os.path.join(self.save_dir, exp.get_plot_name())
    fig = exp.generate_plot(self.viewer_size, index)
    if self.just_view:
      print(exp.get_plot_name())
      plt.show()
    else:
      fig.savefig(os.path.join(self.top_dir, fname))
    # plt.show()
    return latex_string.format(fname[:-4], exp.get_latex_name(), caption)


  def build_latex_stat_table(self):
    stats = []
    for exp in self.experiments:
      stats.append(exp.get_stats_all())
    stats = np.array(stats)
    baselines = np.mean(stats[:,-4:],0)
    latex_string  = "\\begin{table}[H]\n"
    latex_string += "  \\caption{Numerical results}\n"
    latex_string += "      \\begin{tabular}{|l|c|c|c|} \\hline\n"
    latex_string += "                                     "
    latex_string += " &  Same Point Similarities  &  Different Point Similarities  \\\\ \\hline\n"
    latex_string += "Baseline   &"
    latex_string += " {:.2e} $\\pm$ {:.2e} & {:.2e} $\\pm$ {:.2e}".format(*list(baselines))
    latex_string += " \\\\ \\hline\n"
    for exp, stat in zip(self.experiments, stats):
      s = stat[:4]
      latex_string += "{}   &".format(exp.get_name())
      latex_string += " {:.2e} $\\pm$ {:.2e} & {:.2e} $\\pm$ {:.2e}".format(*list(s))
      latex_string += " \\\\ \\hline\n"
    latex_string += "      \\end{tabular}\n"
    latex_string += "      \\label{fig:tab1}\n"
    latex_string += "\\end{table}\n"
    return latex_string

  def build_final_latex(self):
    ##### Get experiments
    experiments = []
    for exp_ in tqdm.tqdm(opts.experiments, disable=not self.verbose):
      experiments.append(Experiment(exp_, verbose=self.verbose))
      
    ##### Get representatives
    losses = [ exp.get_losses()[1][-1] for exp in experiments ]
    min_idx = np.argmin(losses)
    max_idx = np.argmax(losses)
    rnd_idx = min_idx
    if len(losses) > 2:
      while rnd_idx in [ min_idx, max_idx ]:
        rnd_idx = np.random.randint(len(losses))

    ##### Create Latex String
    latex_string = ""
    latex_string += self.save_images_and_output_latex(
                        self.experiments[min_idx], 
                        "Sample embedding, similarity matrix, histogram, and loss curves for best performing of this set", 
                        self.index)
    if len(losses) > 1:
      latex_string += self.save_images_and_output_latex(
                          self.experiments[max_idx],
                          "Sample embedding, similarity matrix, histogram, and loss curves for worst performing of this set", 
                          self.index)
    if len(losses) > 2:
      latex_string += self.save_images_and_output_latex(
                          self.experiments[rnd_idx],
                          "Sample embedding, similarity matrix, histogram, and loss curves for random experiment of this set", 
                          self.index)

    latex_string += self.build_latex_stat_table()
    # latex_string += "\\begin{figure}[H]\n"
    # latex_string += "\\end{figure}\n"
    return latex_string


if __name__ == "__main__":
  # Build options
  # opts = options.get_opts()
  opts = get_latex_opts()
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.serif'] = 'Ubuntu'
  plt.rcParams['font.monospace'] = 'Ubuntu Mono'
  plt.rcParams['font.size'] = 14
  plt.rcParams['axes.labelsize'] = 10
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams['xtick.labelsize'] = 8
  plt.rcParams['ytick.labelsize'] = 8
  plt.rcParams['legend.fontsize'] = 10
  plt.rcParams['figure.titlesize'] = 12
  
  ##### Build Latex
  # ##### Save plots
  # save_dir = '/home/stephen/Downloads'
  # fig = exp.generate_plot(opts.viewer_size)
  # # plt.show()
  # # fig.savefig(os.path.join(save_dir, exp.get_plot_name()))
  # baselines = stats[:,-4:]
  # plt.close(fig)
  if opts.experiments and len(opts.experiments) > 0:
    latex = LatexGenerator(opts)
    print(latex.build_final_latex())
  else:
    if opts.plot_generate == '':
      print("ERROR: No experiments or plot_generate - nothing to do", file=sys.stderr)
      sys.exit(1)
    print()
    exp = Experiment(opts.plot_generate, extention='eps')
    fig = exp.generate_plot(opts.viewer_size, opts.index)
    fname = os.path.join(opts.latex_path, opts.save_dir, exp.get_plot_name())
    fig.savefig(fname)
  # print(baselines.shape)
  # print(np.mean(baselines,0), np.std(baselines,0), np.min(baselines,0), np.max(baselines,0))
  # plt.hist([ baselines[:,0], baselines[:,2] ], 50, histtype='step', range=(0,1), cumulative=True)
  # plt.legend([ 'diag', 'off_diag'])
  # plt.show()
  # plt.hist([ baselines[:,1], baselines[:,3] ], 50, histtype='step', range=(0,0.5), cumulative=True)
  # plt.legend([ 'diag', 'off_diag'])
  # plt.show()





