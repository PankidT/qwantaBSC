import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import dill
# import seaborn as sns
import matplotlib.pyplot as plt

# Define function
def openPickle(folder):
    with open(folder+'/Result_epps_expEPPS_p0_r0.pkl', 'rb') as f:
        file1 = dill.load(f)
    with open(folder+'/Result_epps_expEPPS_p0_r1.pkl', 'rb') as f:
        file2 = dill.load(f)
    with open(folder+'/Result_epps_expEPPS_p0_r2.pkl', 'rb') as f:
        file3 = dill.load(f)
    with open(folder+'/Result_epps_expEPPS_p0_r3.pkl', 'rb') as f:
        file4 = dill.load(f)
    with open(folder+'/Result_epps_expEPPS_p0_r4.pkl', 'rb') as f:
        file5 = dill.load(f)
    all_file = [file1, file2, file3, file4, file5]
    return all_file

def plot_all(dataset, init, final, title):
    fig = plt.figure(figsize=(10, 4))
    x_data = np.linspace(init, final, len(dataset[0]['Fidelity History'][init:]))
    # sns.set_theme(style='darkgrid')
    # sns.lineplot(x=x_data, y=dataset[0]['Fidelity History'][init:final+1])
    # sns.lineplot(x=x_data, y=dataset[1]['Fidelity History'][init:final+1])
    # sns.lineplot(x=x_data, y=dataset[2]['Fidelity History'][init:final+1])
    # sns.lineplot(x=x_data, y=dataset[3]['Fidelity History'][init:final+1])
    # sns.lineplot(x=x_data, y=dataset[4]['Fidelity History'][init:final+1])

    plt.plot(x_data, dataset[0]['Fidelity History'][init:final+1], '.r')
    plt.plot(x_data, dataset[1]['Fidelity History'][init:final+1], '.g')
    plt.plot(x_data, dataset[2]['Fidelity History'][init:final+1], '.b')
    plt.plot(x_data, dataset[3]['Fidelity History'][init:final+1], '.c')
    plt.plot(x_data, dataset[4]['Fidelity History'][init:final+1], '.m')
    plt.xlabel('State Tomography')
    plt.ylabel('Fidelity')
    plt.title(title)
    st.pyplot(fig)

    avg_ssdp = []
    avg_ssdp.append(dataset[0]['Fidelity History'][init:final+1])
    avg_ssdp.append(dataset[1]['Fidelity History'][init:final+1])
    avg_ssdp.append(dataset[2]['Fidelity History'][init:final+1])
    avg_ssdp.append(dataset[3]['Fidelity History'][init:final+1])
    avg_ssdp.append(dataset[4]['Fidelity History'][init:final+1])
    avg_ssdp = np.array(avg_ssdp)

    mean_ssdp = avg_ssdp.mean(axis=0)
    std_ssdp = avg_ssdp.std(axis=0)
    print("9000 state tomography fidelity", round(mean_ssdp[-1], 3))
    plt.xlabel('State Tomography')
    plt.ylabel('Fidelity')
    plt.title(title)

    plt.plot(np.linspace(init, final+1, final+1-init), mean_ssdp, '-')
    plt.fill_between(np.linspace(init, final+1, final+1-init), mean_ssdp-std_ssdp, mean_ssdp+std_ssdp, alpha=0.5)
    plt.show()

    return round(mean_ssdp[-1], 3)

selected = option_menu(
        menu_title = "Main Menu",
        options = ["Fidelity", "Heatmap", "FOA"],
        orientation = "horizontal",
    )

if selected =="Fidelity":
    st.title(f'You have selected {selected}')
elif selected =="Heatmap":
    st.title(f'You have selected {selected}')
elif selected =="1st order approximation":
    st.title(f'You have selected {selected}')

def user_input_features():
    loss = st.select_slider('Loss (dB/km)', options=[0.001, 0.003, 0.006, 0.009])
    tau = st.select_slider('Cohereance time (s)', options=[0.25, 0.5, 1])
    meaErr = st.select_slider('Measurement error', options=[0.01, 0.03, 0.05])
    return loss, tau, meaErr

# use function that design above
loss, tau, meaErr = user_input_features()

if meaErr == 0.01:
    if tau == 1:
        if loss == 0.001:
            f_BKK_CM_1_1_1 = plot_all(openPickle('../meaErr_0.01/tau1/BKK_CM_0.001_tau1_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.001 tau=1 meaErr=0.01')
        elif loss == 0.003:
            f_BKK_CM_3_1_1 = plot_all(openPickle('../meaErr_0.01/tau1/BKK_CM_0.003_tau1_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.003 tau=1 meaErr=0.01')
        elif loss == 0.006:
            f_BKK_CM_6_1_1 = plot_all(openPickle('../meaErr_0.01/tau1/BKK_CM_0.006_tau1_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.006 tau=1 meaErr=0.01')
        elif loss == 0.009:
            f_BKK_CM_9_1_1 = plot_all(openPickle('../meaErr_0.01/tau1/BKK_CM_0.009_tau1_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.009 tau=1 meaErr=0.01')

    elif tau == 0.5:
        if loss == 0.001:
            pass
        elif loss == 0.003:
            pass
        elif loss == 0.006:
            pass
        elif loss == 0.009:
            pass

    elif tau == 0.25:
        if loss == 0.001:
            f_BKK_CM_1_025_1 = plot_all(openPickle('../meaErr_0.01/tau25/BKK_CM_0.001_tau0.25_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.001 tau=0.25 meaErr=0.01')
        elif loss == 0.003:
            f_BKK_CM_3_025_1 = plot_all(openPickle('../meaErr_0.01/tau25/BKK_CM_0.003_tau0.25_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.003 tau=0.25 meaErr=0.01')
        elif loss == 0.006:
            f_BKK_CM_6_025_1 = plot_all(openPickle('../meaErr_0.01/tau25/BKK_CM_0.006_tau0.25_meaErr1'), 1000, 9000, 'BKK_CM, loss=0.006 tau=0.25 meaErr=0.01')
        elif loss == 0.009:
            pass

elif meaErr == 0.03:
    if tau == 1:
        if loss == 0.001:
            f_BKK_CM_1_1_3 = plot_all(openPickle('../tau1/BKK_CM_0.001_tau1'), 1000, 9000, 'BKK_CM, loss=0.001 tau=1 meaErr=0.03')
        elif loss == 0.003:
            f_BKK_CM_3_1_3 = plot_all(openPickle('../tau1/BKK_CM_0.003_tau1'), 1000, 9000, 'BKK_CM, loss=0.003 tau=1 meaErr=0.03')
        elif loss == 0.006:
            f_BKK_CM_6_1_3 = plot_all(openPickle('../tau1/BKK_CM_0.006_tau1'), 1000, 9000, 'BKK_CM, loss=0.006 tau=1 meaErr=0.03')
        elif loss == 0.009:
            f_BKK_CM_9_1_3 = plot_all(openPickle('../tau1/BKK_CM_0.009_tau1'), 1000, 9000, 'BKK_CM, loss=0.009 tau=1 meaErr=0.03')

    elif tau == 0.5:
        if loss == 0.001:
            f_BKK_CM_1_05_3 = plot_all(openPickle('../tau05/BKK_CM_0.001_a'), 1000, 9000, 'BKK_CM, loss=0.001 tau=0.5 meaErr=0.03')
        elif loss == 0.003:
            f_BKK_CM_3_05_3 = plot_all(openPickle('../tau05/BKK_CM_0.003_a'), 1000, 9000, 'BKK_CM, loss=0.003 tau=0.5 meaErr=0.03')
        elif loss == 0.006:
            f_BKK_CM_6_05_3 = plot_all(openPickle('../tau05/BKK_CM_0.006_a'), 1000, 9000, 'BKK_CM, loss=0.006 tau=0.5 meaErr=0.03')
        elif loss == 0.009:
            f_BKK_CM_9_05_3 = plot_all(openPickle('../tau05/BKK_CM_0.009_a'), 1000, 9000, 'BKK_CM, loss=0.009 tau=0.5 meaErr=0.03')

    elif tau == 0.25:
        if loss == 0.001:
            f_BKK_CM_1_025_3 = plot_all(openPickle('../tau25/BKK_CM_0.001_tau.25'), 1000, 9000, 'BKK_CM, loss=0.001 tau=0.25 meaErr=0.03')
        elif loss == 0.003:
            f_BKK_CM_3_025_3 = plot_all(openPickle('../tau25/BKK_CM_0.003_tau.25'), 1000, 9000, 'BKK_CM, loss=0.003 tau=0.25 meaErr=0.03')
        elif loss == 0.006:
            f_BKK_CM_6_025_3 = plot_all(openPickle('../tau25/BKK_CM_0.006_tau.25'), 1000, 9000, 'BKK_CM, loss=0.006 tau=0.25 meaErr=0.03')
        elif loss == 0.009:
            f_BKK_CM_9_025_3 = plot_all(openPickle('../tau25/BKK_CM_0.009_tau.25'), 1000, 9000, 'BKK_CM, loss=0.009 tau=0.25 meaErr=0.03')

elif meaErr == 0.05:
    if tau == 1:
        if loss == 0.001:
            f_BKK_CM_1_1_5 = plot_all(openPickle('../meaErr_0.05/tau1/BKK_CM_0.001_tau1_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.001 tau=1 meaErr=0.05')
        elif loss == 0.003:
            f_BKK_CM_3_1_5 = plot_all(openPickle('../meaErr_0.05/tau1/BKK_CM_0.003_tau1_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.003 tau=1 meaErr=0.05')
        elif loss == 0.006:
            f_BKK_CM_6_1_5 = plot_all(openPickle('../meaErr_0.05/tau1/BKK_CM_0.006_tau1_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.006 tau=1 meaErr=0.05')
        elif loss == 0.009:
            f_BKK_CM_9_1_5 = plot_all(openPickle('../meaErr_0.05/tau1/BKK_CM_0.009_tau1_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.009 tau=1 meaErr=0.05')

    elif tau == 0.5:
        if loss == 0.001:
            pass
        elif loss == 0.003:
            pass
        elif loss == 0.006:
            pass
        elif loss == 0.009:
            pass

    elif tau == 0.25:
        if loss == 0.001:
            f_BKK_CM_1_025_5 = plot_all(openPickle('../meaErr_0.05/tau25/BKK_CM_0.001_tau0.25_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.001 tau=0.25 meaErr=0.05')
        elif loss == 0.003:
            f_BKK_CM_3_025_5 = plot_all(openPickle('../meaErr_0.05/tau25/BKK_CM_0.003_tau0.25_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.003 tau=0.25 meaErr=0.05')
        elif loss == 0.006:
            f_BKK_CM_6_025_5 = plot_all(openPickle('../meaErr_0.05/tau25/BKK_CM_0.006_tau0.25_meaErr5'), 1000, 9000, 'BKK_CM, loss=0.006 tau=0.25 meaErr=0.05')
        elif loss == 0.009:
            pass



instructions = """
Multiselection
"""

place = ['test1', 'test2', 'test3', 'pandas', 'keras']
loss = ['0.001', '0.003', '0.006', '0.009']

select_place = st.multiselect(
    "Select Python packages to compare",
    place,
    default=[
        "pandas",
        "keras"
    ],
    help=instructions
)
if st.checkbox('View logarithmic scale'):
    st.write('test')