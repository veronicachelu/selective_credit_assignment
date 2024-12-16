from imports import *
import pandas as pd


def save_fig_pdf(fig, name):
    file = './plots/{}.pdf'.format(name)
    with open(file, 'wb') as f:
        fig.savefig(f, facecolor='none', edgecolor='none', format='pdf', bbox_inches="tight")

def save_fig_png(fig, name):
    file = './plots/{}.pdf'.format(name)
    with open(file, 'wb') as f:
        fig.savefig(f, facecolor='none', edgecolor='none', format='png', bbox_inches="tight")


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def print_stats_csv(df):
    print(df.columns)

