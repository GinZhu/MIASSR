import matplotlib.pyplot as plt
import torch
from os.path import join, isdir, exists
from os import makedirs, listdir
from glob import glob
from tabulate import tabulate
import numpy as np
from datasets.basic_dataset import BasicCropTransform
from tabulate import tabulate


class Crop(BasicCropTransform):

    def __init__(self, size):
        super(Crop, self).__init__(size, 0)

    def __call__(self, in_img, cx, cy):
        x = int(cx - self.size[0]//2)
        y = int(cy - self.size[1]//2)
        return in_img[x:x+self.size[0], y:y+self.size[1]]


class MetaSRImagePlotter(object):
    def __init__(self, paras):
        self.paras = paras

        self.output_dir = paras.output_dir
        self.root_folder = paras.root_folder
        self.gt_images_folder = paras.gt_images_folder
        self.rec_imgs_folder = paras.rec_images_folder
        self.reports_folder = None

        self.name = paras.name
        self.verbose = paras.verbose

        self.report_postfix = paras.report_postfix
        self.image_postfix = paras.image_postfix

        self.basic_metrics = paras.metrics
        self.sr_factors = paras.sr_scales

        self.patch_size = paras.patch_size
        self.crop_func = Crop(self.patch_size)

        # color bar
        self.color_bar_range = paras.color_bar_range
        self.color_bar_unit = paras.color_bar_unit

        # patients
        self.all_patients_ids = []

        # multi modalities
        if 'multi' in self.name:
            self.multi_modalities = paras.modalities
        else:
            self.multi_modalities = None

    def setup(self):
        plog = self.fancy_print('Plot image of {}, with paras:'.format(self.name))
        plog += '\n' + str(self.paras) + '\n\n'

        self._creat_dirs()

        self.reports_folder = join(self.root_folder, self.rec_imgs_folder, 'reports')
        self.rec_imgs_folder = join(self.root_folder, self.rec_imgs_folder, 'inferences')
        self.gt_images_folder = join(self.root_folder, self.gt_images_folder, 'inferences')

        patients = listdir(self.reports_folder)
        self.all_patients_ids = [_.replace(self.report_postfix, '') for _ in patients]

        self.write_log(plog)

    def set_patch_size(self, size):
        self.patch_size = size
        self.crop_func = Crop(size)

    def _render_with_color_bar(self, diff):
        diff = diff / self.color_bar_range
        rendered_diff = np.concatenate([
            diff * self.color_bar_unit[0],
            np.abs(diff * self.color_bar_unit[1]),
            diff * self.color_bar_unit[2]
        ], axis=-1)

        rendered_diff = np.clip(rendered_diff, 0., 1.)
        rendered_diff = 1 - rendered_diff
        rendered_diff = (rendered_diff * 255).astype('uint8')
        return rendered_diff

    def _color_bar(self):
        color_bar = np.arange(1, -1.0001, -0.01)
        color_bar = color_bar * self.color_bar_range
        color_bar = np.tile(color_bar, [1, 20, 1]).transpose(2, 1, 0)
        color_bar = self._render_with_color_bar(color_bar)
        return color_bar

    @staticmethod
    def _to_rgb(img):
        img = np.concatenate([img, img, img], axis=-1)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype('uint8')
        return img

    @staticmethod
    def _save_image(path, img):
        plt.imsave(path, img)

    def run(self, cx, cy, slice_ids=None):
        if slice_ids is None:
            slice_ids = [50, 60, 70, 80, 90]
        for pid in self.all_patients_ids:
            for sid in slice_ids:
                self.draw(pid, sid, cx, cy)

    def draw(self, pid, sid, cx, cy):
        gt_imgs_path = join(
            self.gt_images_folder,
            '{}{}'.format(pid, self.image_postfix))
        rec_imgs_path = join(
            self.rec_imgs_folder,
            '{}{}'.format(pid, self.image_postfix))

        gt_imgs = self._load_image(gt_imgs_path, sid, gt=True)
        rec_imgs = self._load_image(rec_imgs_path, sid, gt=False)

        ori_image = gt_imgs[self.sr_factors[-1]]

        gt_patches = self._crop(gt_imgs, cx, cy)
        rec_patches = self._crop(rec_imgs, cx, cy)

        diff_patches = {}
        for s in self.sr_factors:
            diff_patches[s] = rec_patches[s] - gt_patches[s]

        # save images
        cod = self.exist_or_make(join(
            self.output_dir, '{}_slice_{}'.format(pid, sid)
        ))
        self._save_image(join(cod, 'color_bar.png'), self._color_bar())
        if self.multi_modalities:
            for i, m in enumerate(self.multi_modalities):
                self._save_image(
                    join(cod, '{}_ori.png'.format(m)),
                    self._to_rgb(ori_image[:, :, i:i+1]))
                for s in self.sr_factors:
                    diff = diff_patches[s][:, :, i:i+1]
                    rec = rec_patches[s][:, :, i:i+1]
                    merged = np.concatenate(
                        [self._to_rgb(rec), self._render_with_color_bar(diff)],
                        axis=1
                    )
                    self._save_image(join(
                        cod, '{}_sr{}.png'.format(m, s)
                    ), merged)

        else:
            ori_image = self._to_rgb(ori_image)
            self._save_image(join(cod, 'ori.png'), ori_image)
            for s in self.sr_factors:
                diff = diff_patches[s]
                rec = rec_patches[s]
                merged = np.concatenate(
                    [self._to_rgb(rec), self._render_with_color_bar(diff)],
                    axis=1
                )
                self._save_image(join(
                    cod, 'sr{}.png'.format(s)
                ), merged)

        # plog
        repo_path = join(
            self.reports_folder,
            '{}{}'.format(pid, self.report_postfix)
        )
        scores = self._load_scores(repo_path, sid)

        plog = self.fancy_print('{} slice {} evaluate scores:'.format(
            pid, sid
        )) + '\n'
        plog += 'Patch at {}\n'.format([cx, cy])
        plog += self.print_scores(scores)

        self.write_log(plog)

    def print_scores(self, scores):
        if self.multi_modalities:
            plog = ''
            for m in self.multi_modalities:
                plog += '{}\n{}\n'.format(m, self._print_one_scores(scores[m]))
            return plog
        else:
            return self._print_one_scores(scores)

    def _print_one_scores(self, scores):
        table = []
        for s in self.sr_factors:
            row = ['{:.2}'.format(s)]
            for m in self.basic_metrics:
                v = scores[s][m]
                row += ['{:.4}'.format(v)]

            table.append(row)
        headers = ['SR', ] + self.basic_metrics
        plog = tabulate(table, headers=headers)
        return plog

    @staticmethod
    def fancy_print(m):
        l = len(m)
        return '#' * (l + 50) + '\n' + '#' * 5 + ' ' * 20 + m + ' ' * 20 + '#' * 5 + '\n' + '#' * (l + 50)

    def _load_image(self, path, sid, gt=False):
        images = torch.load(path)
        if gt:
            images = images['gt_imgs']
        else:
            images = images['rec_imgs']
        return images[sid]

    def _load_scores(self, path, sid):
        repos = torch.load(path)['eva_report']
        if self.multi_modalities:
            scores = {}
            for m in self.multi_modalities:
                scores[m] = self._load_one_scores(repos[m], sid)
            return scores
        else:
            return self._load_one_scores(repos, sid)

    def _load_one_scores(self, repos, sid):
        scores = {}
        for s in self.sr_factors:
            case_s = {}
            for m in self.basic_metrics:
                k = '{}_{}'.format(m, s)
                if m == 'fid':
                    case_s[m] = repos[k][0]
                else:
                    case_s[m] = repos[k][sid]
            scores[s] = case_s
        return scores

    def _crop(self, images, cx, cy):
        patches = {}
        for s in self.sr_factors:
            p = self.crop_func(images[s], cx, cy)
            patches[s] = p
        return patches

    @staticmethod
    def _image_diff(rec_img, gt_img):
        return rec_img - gt_img

    def _creat_dirs(self):
        """
        Testing passed.

        :return:
        """
        self.output_dir = join(self.output_dir, self.name)

        # creat some dirs for output
        self.output_dir = self.exist_or_make(self.output_dir)

        # training log
        self.log = join(self.output_dir, 'draw_log.txt')

    @staticmethod
    def exist_or_make(path):
        if not isdir(path):
            makedirs(path)
        return path

    def write_log(self, plog):
        if self.verbose:
            print(plog)
        with open(self.log, 'a') as f:
            f.write(plog + '\n')


class MetaSRResultPlotter(object):

    def __init__(self, paras):
        self.paras = paras

        self.output_dir = paras.output_dir
        self.root_folder = paras.root_folder
        self.name = paras.name
        self.verbose = paras.verbose

        self.methods = paras.methods
        self.report_folders = {}
        self.report_postfix = paras.report_postfix

        self.basic_metrics = paras.metrics
        self.sr_factors = paras.sr_scales

        self.reports = {}

    def setup(self):
        plog = self.fancy_print('Summary results of {}, with paras:'.format(self.name))
        plog += '\n' + str(self.paras) + '\n\n'

        self._creat_dirs()

        if len(self.methods):
            for m in self.methods:
                self.report_folders[m] = self.paras.report_folders[m]
        else:
            self.report_folders = self.paras.report_folders

        # load all reports
        self._load()

        self.write_log(plog)

    def plot(self):
        for m in self.basic_metrics:
            self._plot_one_metric(m)

    @staticmethod
    def fancy_print(m):
        l = len(m)
        return '#' * (l + 50) + '\n' + '#' * 5 + ' ' * 20 + m + ' ' * 20 + '#' * 5 + '\n' + '#' * (l + 50)

    @staticmethod
    def _load_one_report(path):
        report = torch.load(path)
        return report['eva_report']

    def _load_reports(self, method):
        folder = join(self.root_folder, self.report_folders[method])
        all_report_files = glob(join(folder, self.report_postfix))
        all_reports = []
        for p in all_report_files:
            repo = self._load_one_report(p)
            repo = self.report_filter(repo, self.basic_metrics)
            all_reports.append(repo)

        all_reports = self.stack_eva_reports(all_reports)

        return all_reports

    def _load(self):
        for m in self.methods:
            self.reports[m] = self._load_reports(m)

    def _plot_one_metric(self, metric):
        plog = self.fancy_print('Plotting results of {}'.format(metric.upper)) + '\n'

        eva_means = {}
        for m in self.methods:
            mean, std = self._analysis_one_metric_of_one_method(metric, m)
            eva_means[m] = mean
        # for k in eva_means:
        #     print(k, np.mean(eva_means[k]))

        # plot
        keys, values = zip(*sorted(eva_means.items()))
        values = np.stack(values, axis=-1)

        plog += 'Mean {} of {}:\n {}'.format(metric, keys, np.mean(values, axis=0))

        plt.plot(self.sr_factors, values)
        plt.grid(True)
        plt.xlabel('SR scale', fontsize='x-large')
        plt.ylabel(metric, fontsize='x-large')
        plt.legend(keys, fontsize='x-large')
        plt.savefig(join(self.plots_dir, '{}_{}.png'.format(self.name, metric.upper())))
        plt.close()

        # save in txt
        header = 'SR Scale\t' + '\t'.join(keys)
        values = np.concatenate([np.array(self.sr_factors).reshape(-1, 1), values], axis=-1)

        np.savetxt(join(self.records_dir, '{}.txt'.format(metric.upper())), values,
                   header=header, delimiter='\t', fmt='%.4f')

        self.write_log(plog)

    def _analysis_one_metric_of_one_method(self, metric, method):
        repo = self.reports[method]
        eva_mean = []
        eva_std = []
        for s in self.sr_factors:
            v = repo['{}_{}'.format(metric, s)]
            if isinstance(v[0], list):
                v = np.concatenate(v)
            eva_mean.append(np.mean(v))
            eva_std.append(np.std(v))
        return eva_mean, eva_std

    def print_a_report(self, report):
        table = []
        for s in self.sr_factors:
            row = ['{:.2}'.format(s), ]
            for m in self.basic_metrics:
                v = report['{}_{}'.format(m, s)]
                if isinstance(v, (float, int)):
                    row += ['{:.4}'.format(v)]
                else:
                    if isinstance(v[0], list):
                        v = np.concatenate(v)
                    mean_v = np.mean(v)
                    std_v = np.std(v)
                    row += ['{:.4}({:.2})'.format(mean_v, std_v)]
            table.append(row)
        headers = ['SR', ] + self.basic_metrics
        plog = tabulate(table, headers=headers)
        return plog

    @staticmethod
    def stack_eva_reports(reports):
        # stack each element in eva_report separately
        stacked_report = {}
        for k in reports[0].keys():
            stacked_report[k] = []
            for r in reports:
                stacked_report[k].append(r[k])
        return stacked_report

    @staticmethod
    def report_filter(report, keys):
        if len(keys):
            r = {}
            for k in report:
                if k.split('_')[0] in keys:
                    r[k] = report[k]
            return r
        return report

    def _creat_dirs(self):
        """
        Testing passed.

        :return:
        """
        self.output_dir = join(self.output_dir, self.name)

        # creat some dirs for output
        self.output_dir = self.exist_or_make(self.output_dir)

        self.plots_dir = self.exist_or_make(join(
            self.output_dir, 'plots'))
        self.records_dir = self.exist_or_make(join(
            self.output_dir, 'records'
        ))

        # training log
        self.log = join(self.output_dir, 'plots_log.txt')

    @staticmethod
    def exist_or_make(path):
        if not isdir(path):
            makedirs(path)
        return path

    def write_log(self, plog):
        if self.verbose:
            print(plog)
        with open(self.log, 'a') as f:
            f.write(plog + '\n')


class ModelSizeCounter(object):

    def __init__(self, root_folder):
        self.all_model_paths = glob(join(root_folder, '*.pt'), recursive=True)

        self.model_names = [_.split('/')[-1] for _ in self.all_model_paths]

    def __call__(self, models=[]):
        for m in models:
            # c = 0
            for p, name in zip(self.all_model_paths, self.model_names):
                if m in p:
                    c_one = self.count(p)
                    print('Model {} have {} parameters in total'.format(name, c_one))
                    # c += c_one
            # print('Model {} have {} parameters in total'.format(m, c))

    def count(self, path):
        ptm = torch.load(path, map_location='cpu')
        c = 0
        for n, t in ptm.items():
            c += torch.numel(t)

        return c


class MultiModalityMetaSRResultPlotter(MetaSRResultPlotter):

    def __init__(self, paras, modality):
        super(MultiModalityMetaSRResultPlotter, self).__init__(paras)

        self.modality = modality

        self.name = '{}_{}'.format(self.name, self.modality)

    def _load_one_report(self, path):
        repo = super(MultiModalityMetaSRResultPlotter, self)._load_one_report(path)
        repo = repo[self.modality]
        return repo


