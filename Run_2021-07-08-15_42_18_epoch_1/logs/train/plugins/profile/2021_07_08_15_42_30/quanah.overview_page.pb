?	?v5y?&@?v5y?&@!?v5y?&@	?G?(!???G?(!??!?G?(!??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?v5y?&@?!6X8I??A?H0??*&@YO?S?{F??rEagerKernelExecute 0*	?|?5^jl@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatfKVE?ɰ?!??3
?<@)???Đ???1?$?? ?8@:Preprocessing2U
Iterator::Model::ParallelMapV2Z?h9?C??!???8E2@)Z?h9?C??1???8E2@:Preprocessing2F
Iterator::Model?yUg???!i????A@)׆?q?&??1YA??P1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???뉮??!ջ???'@)???뉮??1ջ???'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	oB@???!K(? ?P@)('?UH???1??n8?P&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ɋL????!???6@)??p?????1D
?Z$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*T7ۃ?!lɫ??@)*T7ۃ?1lɫ??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??R?h??!??3?h8@)???0`?u?1:?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?G?(!??Ix??\{?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?!6X8I???!6X8I??!?!6X8I??      ??!       "      ??!       *      ??!       2	?H0??*&@?H0??*&@!?H0??*&@:      ??!       B      ??!       J	O?S?{F??O?S?{F??!O?S?{F??R      ??!       Z	O?S?{F??O?S?{F??!O?S?{F??b      ??!       JCPU_ONLYY?G?(!??b qx??\{?X@Y      Y@q?4?????"?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 