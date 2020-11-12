from scipy import stats
import xlsxwriter
import os
import numpy as np


def read_file(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()

    data = []
    for i in range(1, len(lines)):
        if i >= 11:
            break
        data.append(float(lines[i].replace('\n', '')))
    return data


def write_win_draw_loss(writer, f, a, d1, d2, p_value, alpha):
    if p_value >= alpha:
        writer.write(f + 2, 2 + 4 * a, 0)
        writer.write(f + 2, 3 + 4 * a, 1)
        writer.write(f + 2, 4 + 4 * a, 0)
    else:
        if np.mean(d1) > np.mean(d2):
            writer.write(f + 2, 2 + 4 * a, 1)
            writer.write(f + 2, 3 + 4 * a, 0)
            writer.write(f + 2, 4 + 4 * a, 0)
        elif np.mean(d1) == np.mean(d2):
            writer.write(f + 2, 2 + 4 * a, 0)
            writer.write(f + 2, 3 + 4 * a, 1)
            writer.write(f + 2, 4 + 4 * a, 0)
        else:
            writer.write(f + 2, 2 + 4 * a, 0)
            writer.write(f + 2, 3 + 4 * a, 0)
            writer.write(f + 2, 4 + 4 * a, 1)

result_dir = './wilcoxon_data'
algo1 = 'OTR'
algo2_dir = ['EWC', 'o-EWC', 'SI', 'LwF', 'GR', 'GR+distill', 'A-GEM', 'ER', 'OTR+distill', 'iCaRL']
alpha = 0.05
workbook = xlsxwriter.Workbook('OTR_wilcoxon.xlsx')
cell_format = workbook.add_format()
cell_format.set_text_wrap()
wilcoxon_worksheet = workbook.add_worksheet("Wilcoxon")
levene_worksheet = workbook.add_worksheet("Levene")

wilcoxon_worksheet.set_column('A:A', 25)
levene_worksheet.set_column('A:A', 25)

# Create a format to use in the merged range.
merge_format = workbook.add_format({
    'bold': 0,
    'border': 1,
    'align': 'center',
    'valign': 'vcenter',
    'fg_color': 'white'})


files = ['splitMNIST_task', 'splitMNIST_domain', 'splitMNIST_class',
         'permMNIST_task', 'permMNIST_domain', 'permMNIST_class',
         'rotMNIST_task', 'rotMNIST_domain', 'rotMNIST_class',
         'CIFAR10_task', 'CIFAR10_domain', 'CIFAR10_class',
         'CIFAR100_task', 'CIFAR100_domain', 'CIFAR100_class']

for a in range(len(algo2_dir)):
    if algo2_dir[a] != 'iCaRL':
        wilcoxon_worksheet.merge_range(0, 1 + 4 * a, 0, 4 + 4 * a, algo2_dir[a], merge_format)
        # levene_worksheet.merge_range(0, 1 + 4 * a, 0, 4 + 4 * a, algo2_dir[a], merge_format)

        wilcoxon_worksheet.write(1, 1 + 4 * a, 'P-value')
        wilcoxon_worksheet.write(1, 2 + 4 * a, 'Win')
        wilcoxon_worksheet.write(1, 3 + 4 * a, 'Draw')
        wilcoxon_worksheet.write(1, 4 + 4 * a, 'Lose')

        # levene_worksheet.write(1, 1 + 4 * a, 'P-value')
        # levene_worksheet.write(1, 2 + 4 * a, 'Win')
        # levene_worksheet.write(1, 3 + 4 * a, 'Draw')
        # levene_worksheet.write(1, 4 + 4 * a, 'Lose')
        for f in range(len(files)):
            try:
                exists1 = os.path.isfile('%s/%s_%s.dat' % (result_dir, algo1, files[f]))
                exists2 = os.path.isfile('%s/%s_%s.dat' % (result_dir, algo2_dir[a], files[f]))

                if exists1 and exists2:
                    d1 = read_file('%s/%s_%s.dat' % (result_dir, algo1, files[f]))
                    d2 = read_file('%s/%s_%s.dat' % (result_dir, algo2_dir[a], files[f]))
                    wil_stat, wil_p_value = stats.wilcoxon(d1, d2)
                    wilcoxon_worksheet.write(f + 2, 0, files[f])
                    wilcoxon_worksheet.write(f + 2, 1 + 4 * a, float(wil_p_value))
                    write_win_draw_loss(wilcoxon_worksheet, f, a, d1, d2, wil_p_value, alpha)

                    # lev_stat, lev_p_value = stats.levene(d1, d2)
                    # levene_worksheet.write(f + 2, 0, files[f])
                    # levene_worksheet.write(f + 2, 1 + 4 * a, float(lev_p_value))
                    #
                    # write_win_draw_loss(levene_worksheet, f, a, d1, d2, lev_p_value, alpha)
                    print(files[f], algo2_dir[a], wil_p_value)
            except Exception as e:
                print(e)

                wilcoxon_worksheet.write(f + 2, 2 + 4 * a, 0)
                wilcoxon_worksheet.write(f + 2, 3 + 4 * a, 1)
                wilcoxon_worksheet.write(f + 2, 4 + 4 * a, 0)
    else:
        for f in range(len(files)):
            if 'class' not in files[f]:
                continue

            wilcoxon_worksheet.merge_range(0, 1 + 4 * a, 0, 4 + 4 * a, algo2_dir[a], merge_format)
            # levene_worksheet.merge_range(0, 1 + 4 * a, 0, 4 + 4 * a, algo2_dir[a], merge_format)

            wilcoxon_worksheet.write(1, 1 + 4 * a, 'P-value')
            wilcoxon_worksheet.write(1, 2 + 4 * a, 'Win')
            wilcoxon_worksheet.write(1, 3 + 4 * a, 'Draw')
            wilcoxon_worksheet.write(1, 4 + 4 * a, 'Lose')

            # levene_worksheet.write(1, 1 + 4 * a, 'P-value')
            # levene_worksheet.write(1, 2 + 4 * a, 'Win')
            # levene_worksheet.write(1, 3 + 4 * a, 'Draw')
            # levene_worksheet.write(1, 4 + 4 * a, 'Lose')

            try:
                exists1 = os.path.isfile('%s/%s_%s.dat' % (result_dir, algo1, files[f]))
                exists2 = os.path.isfile('%s/%s_%s.dat' % (result_dir, algo2_dir[a], files[f]))

                if exists1 and exists2:
                    d1 = read_file('%s/%s_%s.dat' % (result_dir, algo1, files[f]))
                    d2 = read_file('%s/%s_%s.dat' % (result_dir, algo2_dir[a], files[f]))
                    wil_stat, wil_p_value = stats.wilcoxon(d1, d2)
                    wilcoxon_worksheet.write(f + 2, 0, files[f])
                    wilcoxon_worksheet.write(f + 2, 1 + 4 * a, float(wil_p_value))
                    write_win_draw_loss(wilcoxon_worksheet, f, a, d1, d2, wil_p_value, alpha)

                    # lev_stat, lev_p_value = stats.levene(d1, d2)
                    # levene_worksheet.write(f + 2, 0, files[f])
                    # levene_worksheet.write(f + 2, 1 + 4 * a, float(lev_p_value))
                    #
                    # write_win_draw_loss(levene_worksheet, f, a, d1, d2, lev_p_value, alpha)
                    print(files[f], algo2_dir[a], wil_p_value)
            except Exception as e:
                print(e)

                wilcoxon_worksheet.write(f + 2, 2 + 4 * a, 0)
                wilcoxon_worksheet.write(f + 2, 3 + 4 * a, 1)
                wilcoxon_worksheet.write(f + 2, 4 + 4 * a, 0)
workbook.close()