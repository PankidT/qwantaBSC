import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import numpy as np
import dill

# Meta data
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
    sns.set_theme(style='darkgrid')
    sns.lineplot(x=x_data, y=dataset[0]['Fidelity History'][init:final+1])
    sns.lineplot(x=x_data, y=dataset[1]['Fidelity History'][init:final+1])
    sns.lineplot(x=x_data, y=dataset[2]['Fidelity History'][init:final+1])
    sns.lineplot(x=x_data, y=dataset[3]['Fidelity History'][init:final+1])
    sns.lineplot(x=x_data, y=dataset[4]['Fidelity History'][init:final+1])

    # plt.plot(x, dataset[0]['Fidelity History'][init:final+1], '.r')
    # plt.plot(x, dataset[1]['Fidelity History'][init:final+1], '.g')
    # plt.plot(x, dataset[2]['Fidelity History'][init:final+1], '.b')
    # plt.plot(x, dataset[3]['Fidelity History'][init:final+1], '.c')
    # plt.plot(x, dataset[4]['Fidelity History'][init:final+1], '.m')
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

# f_BKK_CM_1_tau1 = plot_all(openPickle('tau1/BKK_CM_0.001_tau1'), 1000, 9000, 'BKK_CM, loss=0.001 tau=1')
# f_BKK_CM_3_tau1 = plot_all(openPickle('tau1/BKK_CM_0.003_tau1'), 1000, 9000, 'BKK_CM, loss=0.003 tau=1')
# f_BKK_CM_6_tau1 = plot_all(openPickle('tau1/BKK_CM_0.006_tau1'), 1000, 9000, 'BKK_CM, loss=0.006 tau=1') ###
# f_BKK_CM_9_tau1 = plot_all(openPickle('tau1/BKK_CM_0.009_tau1'), 1000, 9000, 'BKK_CM, loss=0.009 tau=1')
# f_BKK_CM_12_tau1 = plot_all(openPickle('tau1/BKK_CM_0.012_tau1'), 1000, 9000, 'BKK_CM, loss=0.012 tau=1')

# print('-------------------------------------------------------------------------------------------')

# f_BKK_SK_1_tau1 = plot_all(openPickle('tau1/BKK_SK_0.001_tau1'), 1000, 9000, 'BKK_SK, loss=0.001 tau=1')
# f_BKK_SK_3_tau1 = plot_all(openPickle('tau1/BKK_SK_0.003_tau1'), 1000, 9000, 'BKK_SK, loss=0.003 tau=1')
# f_BKK_SK_6_tau1 = plot_all(openPickle('tau1/BKK_SK_0.006_tau1'), 1000, 9000, 'BKK_SK, loss=0.006 tau=1')
# f_BKK_SK_9_tau1 = plot_all(openPickle('tau1/BKK_SK_0.009_tau1'), 1000, 9000, 'BKK_SK, loss=0.009 tau=1')
# f_BKK_SK_12_tau1 = plot_all(openPickle('tau1/BKK_SK_0.012_tau1'), 1000, 9000, 'BKK_SK, loss=0.012 tau=1')

# print('-------------------------------------------------------------------------------------------')

# f_SK_CM_1_tau1 = plot_all(openPickle('tau1/SK_CM_0.001_tau1'), 1000, 9000, 'SK_CM, loss=0.001 tau=1')
# f_SK_CM_3_tau1 = plot_all(openPickle('tau1/SK_CM_0.003_tau1'), 1000, 9000, 'SK_CM, loss=0.003 tau=1')
# f_SK_CM_6_tau1 = plot_all(openPickle('tau1/SK_CM_0.006_tau1'), 1000, 9000, 'SK_CM, loss=0.006 tau=1')
# f_SK_CM_9_tau1 = plot_all(openPickle('tau1/SK_CM_0.009_tau1'), 1000, 9000, 'SK_CM, loss=0.009 tau=1')

# -----------------------------------------------------------------------------------------------------------------------

st.write('''
# This is multiselection part
''')

instructions = """
Multiselection
"""

package_names = ['test1', 'test2', 'test3', 'pandas', 'keras']

select_packages = st.multiselect(
    "Select Python packages to compare",
    package_names,
    default=[
        "pandas",
        "keras"
    ],
    help=instructions
)
if st.checkbox('View logarithmic scale'):
    st.write('test')

f_BKK_CM_1_tau1 = plot_all(openPickle('../tau1/BKK_CM_0.001_tau1'), 1000, 9000, 'BKK_CM, loss=0.001 tau=1')


st.write('''
# col1, col2
''')

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
            "Select start date",
            date(2020, 1, 1),
            min_value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
            max_value=datetime.now(),
        )

with col2:
    time_frame = st.selectbox(
            "Select weekly or monthly downloads", ("weekly", "monthly")
        )

# def plot(
#     source, x='date', y='downloads', group='project', axis_scale='linear'
# )
#     if st.checkbox('View logarithmic scale'):
#         axis_scale = 'log'

# def main():
#     st.altair_chart(plot_all_downloads)