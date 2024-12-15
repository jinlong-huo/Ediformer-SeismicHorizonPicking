import segyio
import numpy as np
import obspy
import os


# freq_data = None
# with segyio.open('E:\MIAS\F3_crop_horizon_Freq.sgy', mode='r+', strict=True, ignore_geometry=True) as f:
#     for trace in f.trace:
#         temp = trace
#         # temp = torch.tensor(temp)
#         if freq_data is None:
#             freq_data = np_extend(freq_data, temp)
#         else:
#             freq_data = np.append(freq_data, temp)
#             freq_data = torch.Tensor(freq_data)
#             freq_data = freq_data.reshape(-1, 288)
#             print(freq_data.shape)
#     np.save('DemoFreq.npy', freq_data)
# print("Freq convert done!")


def sgy_to_npy(sgy_file, npy_file):
    # Read the SEG-Y file using ObsPy
    st = obspy.read(sgy_file)
    
    # Assuming the SEG-Y file has one trace per channel (standard case)
    # Extract the data as a numpy array (shape: (num_traces, num_samples))
    data = np.array([trace.data for trace in st])

    # Save the data to a .npy file
    np.save(npy_file, data)

    print(f"Conversion complete. Data saved as {npy_file}")

# Example usage:
sgy_file = '/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.sgy'
npy_file = 'F3_amp.npy'

# sgy_to_npy(sgy_file, npy_file)


# with segyio.open('/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.sgy',"r+",ignore_geometry = True) as segyfile:
#     gx=np.zeros(shape=(571551, 288))
#     for i in range(571551):
#         gx[i, :]=segyfile.trace[i]
#     # gx=np.reshape(gx,newshape=(601, 951, 288))#分别对应三维数据体的(inline，xline，time)192  608   448
# gx=np.array(gx)
# np.save('F3_RMSAmp.npy', gx)
# a = np.load(r'F3_RMSAmp.npy')
# print(a.shape)



# def divide_segy_by_inline(input_file, output_directory):
#     # Open the SEG-Y file using segyio
#     with segyio.open(input_file, "r+",ignore_geometry = True) as src:
#         # Access the inline information from the trace headers
#         for i in range(571551):
#             inlines = sorted(set(src.trace))

#         # Create output directory if it doesn't exist
#         if not os.path.exists(output_directory):
#             os.makedirs(output_directory)

#         # Iterate through each inline and save the corresponding traces
#         for inline in inlines:
#             # Extract traces for the current inline
#             inline_traces = [trace for trace in src.trace if trace.raw[:] == inline]

#             # Create a new SEG-Y file for the inline
#             output_file = os.path.join(output_directory, f"F3_amp{inline}.sgy")
#             with segyio.create(output_file, src.bin, src.header) as dst:
#                 for trace in inline_traces:
#                     # Write each trace to the new SEG-Y file
#                     dst.trace[trace.trace_number] = trace.raw
                    
# if __name__ == "__main__":
#     # Replace 'input_file.sgy' with the path to your SEG-Y file
#     input_file = '/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.sgy'

#     # Replace 'output_directory' with the path where you want to save the divided inline sections
#     output_directory = '/home/dell/disk1/Jinlong/Horizontal-data'

#     divide_segy_by_inline(input_file, output_directory)




# with segyio.open(r"D:\Pycharm Projects\1_Original_Seismics.sgy","r+",ignore_geometry = True) as segyfile:
#     gx = np.zeros(shape=(571551, 288))
#     for i in range(571551):
#         gx[i, :]=segyfile.trace[i]
#     # gx=np.reshape(gx,newshape=(601, 951, 288))#分别对应三维数据体的(inline，xline，time)192  608   448
# gx=np.array(gx)
# np.save('F3_RMSAmp.npy', gx)
# a = np.load(r'F3_RMSAmp.npy')
# print(a.shape)

