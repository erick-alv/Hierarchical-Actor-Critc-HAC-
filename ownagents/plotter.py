import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def main():

    df = pd.read_csv('./log_files/2020-05-10-09:19:25_HAC_log.csv')
    print(df.cumulated_reward)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(df.cumulated_reward)), df.cumulated_reward)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # For the minor ticks, use no labels; default NullFormatter.
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.show()

    df = pd.read_csv('./log_files/2020-05-10-09:19:25_other_HAC_log.csv')
    print(df.cumulated_reward)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(df.steps), df.cumulated_reward)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # For the minor ticks, use no labels; default NullFormatter.
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.show()


    '''plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    plt.show()
    plt.plot(r.progress.total_timesteps, r.progress.eprewmean)
    plt.show()'''
    
if __name__ == '__main__':
    main()