"?s
BHostIDLE"IDLE1R??k??@AR??k??@a^????P??i^????P???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1D?l?;ɬ@9D?l?;ɬ@AD?l?;ɬ@ID?l?;ɬ@a???ڦ?i?M?3????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1???M"??@9???M"??@A???M"??@I???M"??@a>???;??i?[?#?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1B`??◡@9B`??◡@AB`??◡@IB`??◡@a??????i@?lj????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a7Z??3???iW?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1?O??nk?@9?O??nk?@A?O??nk?@I?O??nk?@a??M???i
?I?R????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ףp=
??@9ףp=
??@Aףp=
??@Iףp=
??@a ??Q??i?^<?T???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1D?l???@9D?l???@AD?l???@ID?l???@a?@????i???????Unknown
?	HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??v???@9??v???@A??v???@I??v???@aL??T??|?i?`?/ ????Unknown

HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1??C?Nr@9??C?Nr@A??C?Nr@I??C?Nr@a????m?inn????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1P??n?p@9P??n?p@AP??n?p@IP??n?p@a?c???@j?i?SM?Q???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?S㥛?j@9?S㥛?j@A?S㥛?j@I?S㥛?j@aR???e#e?i?9?Wu0???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1??C?l+h@9??C?l+h@A??C?l+h@I??C?l+h@aP?^??/c?ih?yB?C???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1??"???g@9??"???g@A??"???g@I??"???g@a?q???b?im	ɋV???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1?Zd?a@9?Zd?a@A?Zd?a@I?Zd?a@a?_???[?i?ZvLyd???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?Q??a@9?Q??a@A?Q??a@I?Q??a@a?2???/[?i??S-r???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1??~j?LX@9??~j?LX@A??~j?LX@I??~j?LX@aл?q\JS?i??[?{???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?Q??3U@9?Q??3U@A?Q??3U@I?Q??3U@a??=???P?i??b? ????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(15^?IZQ@95^?IZQ@AZd;?OMN@IZd;?OMN@a???-H?i???;$????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1bX9?HN@9bX9?HN@AbX9?HN@IbX9?HN@a???p?
H?i709?&????Unknown?
?HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1R???!K@9R???!K@AR???!K@IR???!K@aFy?I?E?i??`?????Unknown
dHostDataset"Iterator::Model(1q=
ף?V@9q=
ף?V@A#??~j\H@I#??~j\H@a)?}
?VC?ivBN_????Unknown
`HostGatherV2"
GatherV2_1(1?A`??BG@9?A`??BG@A?A`??BG@I?A`??BG@amw?]BwB?i????????Unknown
aHostCast"sequential/Cast(1??x?&QF@9??x?&QF@A??x?&QF@I??x?&QF@a???i?A?i8V?j????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??/ݔE@9??/ݔE@A??/ݔE@I??/ݔE@a?q̕?!A?i/?{;?????Unknown
^HostGatherV2"GatherV2(1?V-E@9?V-E@A?V-E@I?V-E@aB???@?ip?|?????Unknown
iHostWriteSummary"WriteSummary(1u?V>B@9u?V>B@Au?V>B@Iu?V>B@a?	$#?<?i???????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1%??C?L@9%??C?L@A???S??A@I???S??A@aN5?հ?;?i???w????Unknown
`HostGatherV2"
GatherV2_2(1??v??@@9??v??@@A??v??@@I??v??@@a?旘??9?i????5????Unknown
gHostStridedSlice"strided_slice(1q=
ף??@9q=
ף??@Aq=
ף??@Iq=
ף??@a???[9?i?ma????Unknown
wHostReadVariableOp"div_no_nan_2/ReadVariableOp(1??|?5?=@9??|?5?=@A??|?5?=@I??|?5?=@a8g?,?7?i ??pQ????Unknown
? HostSum"Omean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/Sum(1?&1?|:@9?&1?|:@A?&1?|:@I?&1?|:@a?v?5?i??SL?????Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1?n???6@9?n???6@A?n???6@I?n???6@auU??#?1?i????.????Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?I+6@9?I+6@A?I+6@I?I+6@au?w9?|1?i?㎆^????Unknown
?#HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1o???a5@9o???a5@Ao???a5@Io???a5@a??tbe?0?i?2;?}????Unknown
T$HostSub"sub(1X9??v^5@9X9??v^5@AX9??v^5@IX9??v^5@a?????0?i??\??????Unknown
v%HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1\???(|3@9\???(|3@A\???(|3@I\???(|3@a?޿?.?i?Z??????Unknown
V&HostSum"Sum_2(1P??nc2@9P??nc2@AP??nc2@IP??nc2@a͇L?v1-?i?~Ğ^????Unknown
l'HostIteratorGetNext"IteratorGetNext(1????K2@9????K2@A????K2@I????K2@a³?0'?,?iB?71*????Unknown
Y(HostPow"Adam/Pow(1????31@9????31@A????31@I????31@a?7??8P+?i???4?????Unknown
[)HostCast"	Adam/Cast(1?MbX90@9?MbX90@A?MbX90@I?MbX90@a_5?M?)?i?w?Y{????Unknown
?*HostDynamicStitch"Ygradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/DynamicStitch(1?Q???/@9?Q???/@A?Q???/@I?Q???/@a?	O?1$)?iל??????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1?????-@9?????-@A?????-@I?????-@a?ߒ	{'?i6mM?????Unknown
?,HostTile"`gradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/Tile_1(1??C?l+@9??C?l+@A??C?l+@I??C?l+@a??J?%?iog??????Unknown
V-HostSum"Sum_3(1??/?d+@9??/?d+@A??/?d+@I??/?d+@a???2?%?i?=?=????Unknown
?.HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(17?A`?P+@97?A`?P+@A7?A`?P+@I7?A`?P+@av?\X?%?ij;???????Unknown
x/HostDataset"#Iterator::Model::ParallelMapV2::Zip(1H?z??a@9H?z??a@A%??C)@I%??C)@aB?j؜?#?i??????Unknown
?0HostMul"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/mul/Mul(1V-??(@9V-??(@AV-??(@IV-??(@ac?4???#?i]ŦN????Unknown
?1HostAbs"Cmean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1(1??ʡE?'@9??ʡE?'@A??ʡE?'@I??ʡE?'@au?Σ?#?iG1??????Unknown
?2HostMaximum"Emean_absolute_percentage_error/mean_absolute_percentage_error/Maximum(1X9??v?&@9X9??v?&@AX9??v?&@IX9??v?&@aE???0"?i#<??`????Unknown
?3HostSign"Vgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1/Sign(1L7?A`?%@9L7?A`?%@AL7?A`?%@IL7?A`?%@aO|;D/!?i?~s????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1Zd;?O%@9Zd;?O%@AZd;?O%@IZd;?O%@a??Q?+? ?i?t?0?????Unknown
w5HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ˡE???$@9ˡE???$@AˡE???$@IˡE???$@aa??7#? ?iQ??Ҍ????Unknown
p6HostSquaredDifference"SquaredDifference(1q=
ףp$@9q=
ףp$@Aq=
ףp$@Iq=
ףp$@a?I??9 ?iF4=r?????Unknown
[7HostAddV2"Adam/add(1??"??>$@9??"??>$@A??"??>$@I??"??>$@a?{Ň ?i????????Unknown
?8HostDivNoNan"2mean_absolute_percentage_error/weighted_loss/value(1???Q8$@9???Q8$@A???Q8$@I???Q8$@a???? ?i??n?????Unknown
?9HostSub"Amean_absolute_percentage_error/mean_absolute_percentage_error/sub(1?S㥛D#@9?S㥛D#@A?S㥛D#@I?S㥛D#@a??Ѝ???i)z+?????Unknown
e:Host
LogicalAnd"
LogicalAnd(1
ףp=
#@9
ףp=
#@A
ףp=
#@I
ףp=
#@a㌿??:?i%?y????Unknown?
v;HostAssignAddVariableOp"AssignAddVariableOp_4(1?ʡE??"@9?ʡE??"@A?ʡE??"@I?ʡE??"@a9;?i?q??i????Unknown
`<HostDivNoNan"
div_no_nan(1??S???"@9??S???"@A??S???"@I??S???"@a?<?-???i?.BY????Unknown
?=HostRealDiv"Emean_absolute_percentage_error/mean_absolute_percentage_error/truediv(1??|?5?!@9??|?5?!@A??|?5?!@I??|?5?!@a??Q??^?i?aK6<????Unknown
?>HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1-?????!@9-?????!@A-?????!@I-?????!@aj?9?i??A?????Unknown
??HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1B`??"?!@9B`??"?@AB`??"?!@IB`??"?@a v???i??p??????Unknown
V@HostMean"Mean(1????̌!@9????̌!@A????̌!@I????̌!@a??
?B??i?????????Unknown
vAHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1-???g!@9-???g!@A-???g!@I-???g!@a?z???i?qd??????Unknown
?BHostMul"Amean_absolute_percentage_error/mean_absolute_percentage_error/mul(1ffffff!@9ffffff!@Affffff!@Iffffff!@ac-*K??i?????????Unknown
?CHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?????!@9?????!@A?????!@I?????!@aD>??[&?i????o????Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1???S?? @9???S?? @A???S?? @I???S?? @a?f??@??ii??vF????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1\???(? @9\???(? @A\???(? @I\???(? @a\?????i???????Unknown
XFHostMean"Mean_1(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a???????i>-wZ?????Unknown
nGHostMul"Adam/ExponentialDecay/truediv(1??"??~@9??"??~@A??"??~@I??"??~@a9
[?`j?i`~??????Unknown
bHHostDivNoNan"div_no_nan_1(1+?Y@9+?Y@A+?Y@I+?Y@a6??
ML?i???X????Unknown
?IHostAbs"Amean_absolute_percentage_error/mean_absolute_percentage_error/Abs(1??n?@@9??n?@@A??n?@@I??n?@@a?x?m?i?zj|????Unknown
~JHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?(\???@9?(\???@A?(\???@I?(\???@a???82?i[1?????Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_6(1     ?@9     ?@A     ?@I     ?@aW%D޼??i;M?k????Unknown
?LHostDivNoNan"jgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/value/div_no_nan(1?(\??u@9?(\??u@A?(\??u@I?(\??u@a?+??'6?i??Ue????Unknown
[MHostPow"
Adam/Pow_1(11?Zd@91?Zd@A1?Zd@I1?Zd@aP???U(?i?'??????Unknown
?NHostMaximum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Maximum(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a=Jb?
??i?JY?5????Unknown
TOHostAbs"Abs(1??????@9??????@A??????@I??????@ak?6??i?Jٹ????Unknown
vPHostAssignAddVariableOp"AssignAddVariableOp_8(1?rh??|@9?rh??|@A?rh??|@I?rh??|@a;?q?C?i????;????Unknown
oQHostReadVariableOp"Adam/ReadVariableOp(1?v???@9?v???@A?v???@I?v???@a?>?@?i?????????Unknown
?RHostBroadcastTo"Wgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/BroadcastTo(1y?&1?@9y?&1?@Ay?&1?@Iy?&1?@a?C???;?i???5????Unknown
VSHostCast"Cast(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@aiG?yH~?i?????????Unknown
bTHostDivNoNan"div_no_nan_2(1333333@9333333@A333333@I333333@a?Us?t??i?w?w????Unknown
bUHostDivNoNan"div_no_nan_3(1V-?@9V-?@AV-?@IV-?@ap????it??ُ????Unknown
jVHostPow"Adam/ExponentialDecay/Pow(1?????M@9?????M@A?????M@I?????M@am??(Fy?ic}??????Unknown
?WHostFloorDiv"Tgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/floordiv(1q=
ף?@9q=
ף?@Aq=
ף?@Iq=
ף?@a??.S?
?ig7PTi????Unknown
?XHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1+????N@9+????N@A1?Z?@I1?Z?@a?ۭ??
?i??????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a?k,??z	?i????:????Unknown
wZHostReadVariableOp"div_no_nan_3/ReadVariableOp(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a?n
?'	?i@?&?????Unknown
v[HostAssignAddVariableOp"AssignAddVariableOp_7(1?&1?@9?&1?@A?&1?@I?&1?@a?h?˲?iz?9?????Unknown
?\HostDivNoNan"Qmean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/value(1V-?@9V-?@AV-?@IV-?@aE???i?[R=`????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1B`??"?@9B`??"?@AB`??"?@IB`??"?@azZ?%?i?󪱸????Unknown
]^HostCast"Adam/Cast_1(1?l????
@9?l????
@A?l????
@I?l????
@a)K?_?c?is?@????Unknown
T_HostMul"Mul(1L7?A`?	@9L7?A`?	@AL7?A`?	@IL7?A`?	@avo%Î?i???{`????Unknown
?`HostRealDiv"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv(1㥛? ?	@9㥛? ?	@A㥛? ?	@I㥛? ?	@aV??}d?i?3??????Unknown
XaHostCast"Cast_1(1???K7?@9???K7?@A???K7?@I???K7?@a?sU?_z?iF?%??????Unknown
vbHostAssignAddVariableOp"AssignAddVariableOp_5(1?t?V@9?t?V@A?t?V@I?t?V@a?Χc?Q?i?+>M????Unknown
vcHostAssignAddVariableOp"AssignAddVariableOp_1(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a?
??e??i????????Unknown
XdHostCast"Cast_2(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a52??c??i2?5??????Unknown
yeHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1?Q???@9?Q???@A?Q???@I?Q???@a?۽h?f?i)(#X ????Unknown
?fHostCast"]mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/num_elements/Cast(1??????@9??????@A??????@I??????@a??A?%?i0m??d????Unknown
?gHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@a!?_?U ?i ?PE?????Unknown
?hHostNeg"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/sub/Neg(1+????@9+????@A+????@I+????@a????>i s??????Unknown
?iHostSum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/mul/Sum(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a?vL??>i8??$????Unknown
?jHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1?"??~j@9?"??~j@A?"??~j@I?"??~j@a%["@=?>i?5?H_????Unknown
ykHostReadVariableOp"div_no_nan_3/ReadVariableOp_1(1?C?l??@9?C?l??@A?C?l??@I?C?l??@aR??????>io?c?????Unknown
?lHostRealDiv"[gradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv/RealDiv(1?G?z@9?G?z@A?G?z@I?G?z@aڶ{:?>i eT??????Unknown
?mHostMul"Ugradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1/mul(1`??"?? @9`??"?? @A`??"?? @I`??"?? @a??????>i?>?????Unknown
ynHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?E??????9?E??????A?E??????I?E??????a?pt?L?>i?k5"/????Unknown
?oHostSum"Wgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv/Sum(11?Zd??91?Zd??A1?Zd??I1?Zd??aP???U(?>i?A?rW????Unknown
?pHostMean"Bmean_absolute_percentage_error/mean_absolute_percentage_error/Mean(15^?I??95^?I??A5^?I??I5^?I??ad?,?K??>i?y'????Unknown
fqHostMul"Adam/ExponentialDecay(1!?rh????9!?rh????A!?rh????I!?rh????a&0???3?>i?p???????Unknown
?rHostCast"Pgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Cast(1?S㥛???9?S㥛???A?S㥛???I?S㥛???a?ǃ?>iqxǴ?????Unknown
asHostIdentity"Identity(1bX9????9bX9????AbX9????IbX9????a???Q???>ijʚ??????Unknown?
?tHostSum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/sub/Sum(1y?&1???9y?&1???Ay?&1???Iy?&1???a???5ex?>i?????????Unknown*?r
qHost_FusedMatMul"sequential/dense_1/Relu(1D?l?;ɬ@9D?l?;ɬ@AD?l?;ɬ@ID?l?;ɬ@aC@lx?X??iC@lx?X???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1???M"??@9???M"??@A???M"??@I???M"??@a??????i~??G????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1B`??◡@9B`??◡@AB`??◡@IB`??◡@a֙ח????i?hU?Hx???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a!??쏽?i??JF*???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1?O??nk?@9?O??nk?@A?O??nk?@I?O??nk?@al(NKN2??i??:/k???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ףp=
??@9ףp=
??@Aףp=
??@Iףp=
??@aG?<?K??i?V?@%???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1D?l???@9D?l???@AD?l???@ID?l???@awb?????i?l7?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??v???@9??v???@A??v???@I??v???@a}??HF)??i?K???????Unknown
	HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1??C?Nr@9??C?Nr@A??C?Nr@I??C?Nr@aD???d??iD?Y??;???Unknown
?
HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1P??n?p@9P??n?p@AP??n?p@IP??n?p@a??W͞j??iP?8d????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?S㥛?j@9?S㥛?j@A?S㥛?j@I?S㥛?j@a>R?q?K??iT?V?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1??C?l+h@9??C?l+h@A??C?l+h@I??C?l+h@a?ڱ????i?he6jz???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1??"???g@9??"???g@A??"???g@I??"???g@a??D???iS?u?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1?Zd?a@9?Zd?a@A?Zd?a@I?Zd?a@a??U"????i??S????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?Q??a@9?Q??a@A?Q??a@I?Q??a@a?̭^;D??i??yA?X???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1??~j?LX@9??~j?LX@A??~j?LX@I??~j?LX@a??WF?w?i!?VɆ???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?Q??3U@9?Q??3U@A?Q??3U@I?Q??3U@a???U$t?ij??X????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(15^?IZQ@95^?IZQ@AZd;?OMN@IZd;?OMN@a?????l?i!k2??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1bX9?HN@9bX9?HN@AbX9?HN@IbX9?HN@a??>???l?i??w?????Unknown?
?HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1R???!K@9R???!K@AR???!K@IR???!K@a???Fr?i?i?wV?e???Unknown
dHostDataset"Iterator::Model(1q=
ף?V@9q=
ף?V@A#??~j\H@I#??~j\H@a??R?o$g?in??X????Unknown
`HostGatherV2"
GatherV2_1(1?A`??BG@9?A`??BG@A?A`??BG@I?A`??BG@a??~E?f?i0I(E?/???Unknown
aHostCast"sequential/Cast(1??x?&QF@9??x?&QF@A??x?&QF@I??x?&QF@a??'+Y3e?iqS??D???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??/ݔE@9??/ݔE@A??/ݔE@I??/ݔE@a$Pn?z?d?in?WY???Unknown
^HostGatherV2"GatherV2(1?V-E@9?V-E@A?V-E@I?V-E@a??ZU?d?i:k?tm???Unknown
iHostWriteSummary"WriteSummary(1u?V>B@9u?V>B@Au?V>B@Iu?V>B@aƶ?$?Ta?i?????~???Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1%??C?L@9%??C?L@A???S??A@I???S??A@a3
O???`?i?IAW~????Unknown
`HostGatherV2"
GatherV2_2(1??v??@@9??v??@@A??v??@@I??v??@@a
TDWƘ^?i??l?ʞ???Unknown
gHostStridedSlice"strided_slice(1q=
ף??@9q=
ף??@Aq=
ף??@Iq=
ף??@a?"S?W^?i?q??????Unknown
wHostReadVariableOp"div_no_nan_2/ReadVariableOp(1??|?5?=@9??|?5?=@A??|?5?=@I??|?5?=@a?4dZ?"\?i????????Unknown
?HostSum"Omean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/Sum(1?&1?|:@9?&1?|:@A?&1?|:@I?&1?|:@a?'pw)Y?i?[I??????Unknown
? HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1?n???6@9?n???6@A?n???6@I?n???6@ap?jU?iu?׾Q????Unknown
?!HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?I+6@9?I+6@A?I+6@I?I+6@a?r{??T?i.?6G?????Unknown
?"HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1o???a5@9o???a5@Ao???a5@Io???a5@a'b?OT?i??gB?????Unknown
T#HostSub"sub(1X9??v^5@9X9??v^5@AX9??v^5@IX9??v^5@a??	?LT?i???????Unknown
v$HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1\???(|3@9\???(|3@A\???(|3@I\???(|3@avr#ܟ?R?iǓ??W????Unknown
V%HostSum"Sum_2(1P??nc2@9P??nc2@AP??nc2@IP??nc2@aS?)?wQ?i?(g????Unknown
l&HostIteratorGetNext"IteratorGetNext(1????K2@9????K2@A????K2@I????K2@aA?ߜ/Q?i??֓????Unknown
Y'HostPow"Adam/Pow(1????31@9????31@A????31@I????31@a??0?WP?i???g????Unknown
[(HostCast"	Adam/Cast(1?MbX90@9?MbX90@A?MbX90@I?MbX90@a?x?%$?N?i?Q?0????Unknown
?)HostDynamicStitch"Ygradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/DynamicStitch(1?Q???/@9?Q???/@A?Q???/@I?Q???/@a?????N?i|??$???Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1?????-@9?????-@A?????-@I?????-@a??W-L?ire.?+???Unknown
?+HostTile"`gradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/Tile_1(1??C?l+@9??C?l+@A??C?l+@I??C?l+@am8KXVJ?i@x?M?1???Unknown
V,HostSum"Sum_3(1??/?d+@9??/?d+@A??/?d+@I??/?d+@a????
J?i?l?8???Unknown
?-HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(17?A`?P+@97?A`?P+@A7?A`?P+@I7?A`?P+@a?tC??I?i?)??>???Unknown
x.HostDataset"#Iterator::Model::ParallelMapV2::Zip(1H?z??a@9H?z??a@A%??C)@I%??C)@a>??o??G?i?$j7?D???Unknown
?/HostMul"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/mul/Mul(1V-??(@9V-??(@AV-??(@IV-??(@aMZ?7sbG?i?8?dJ???Unknown
?0HostAbs"Cmean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1(1??ʡE?'@9??ʡE?'@A??ʡE?'@I??ʡE?'@aE??f?F?i????P???Unknown
?1HostMaximum"Emean_absolute_percentage_error/mean_absolute_percentage_error/Maximum(1X9??v?&@9X9??v?&@AX9??v?&@IX9??v?&@a?p?!1?E?iR_3z|U???Unknown
?2HostSign"Vgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1/Sign(1L7?A`?%@9L7?A`?%@AL7?A`?%@IL7?A`?%@a?֫^*?D?iH
˄?Z???Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1Zd;?O%@9Zd;?O%@AZd;?O%@IZd;?O%@a	uQ[?>D?i???@?_???Unknown
w4HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ˡE???$@9ˡE???$@AˡE???$@IˡE???$@a?????C?iK?^??d???Unknown
p5HostSquaredDifference"SquaredDifference(1q=
ףp$@9q=
ףp$@Aq=
ףp$@Iq=
ףp$@a?x???jC?i??Q@?i???Unknown
[6HostAddV2"Adam/add(1??"??>$@9??"??>$@A??"??>$@I??"??>$@a:W?ձ;C?i??,Vn???Unknown
?7HostDivNoNan"2mean_absolute_percentage_error/weighted_loss/value(1???Q8$@9???Q8$@A???Q8$@I???Q8$@a5??#_5C?i`???#s???Unknown
?8HostSub"Amean_absolute_percentage_error/mean_absolute_percentage_error/sub(1?S㥛D#@9?S㥛D#@A?S㥛D#@I?S㥛D#@aV?&??MB?i???w???Unknown
e9Host
LogicalAnd"
LogicalAnd(1
ףp=
#@9
ףp=
#@A
ףp=
#@I
ףp=
#@ax??BgB?i?Eߔ<|???Unknown?
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1?ʡE??"@9?ʡE??"@A?ʡE??"@I?ʡE??"@a?cx? B?i??ռ????Unknown
`;HostDivNoNan"
div_no_nan(1??S???"@9??S???"@A??S???"@I??S???"@aU&hd$?A?i??^7????Unknown
?<HostRealDiv"Emean_absolute_percentage_error/mean_absolute_percentage_error/truediv(1??|?5?!@9??|?5?!@A??|?5?!@I??|?5?!@a?Ovqa?@?iI{?u????Unknown
?=HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1-?????!@9-?????!@A-?????!@I-?????!@a\J@??@?i\+?v?????Unknown
?>HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1B`??"?!@9B`??"?@AB`??"?!@IB`??"?@a??????@?iM^i?ܑ???Unknown
V?HostMean"Mean(1????̌!@9????̌!@A????̌!@I????̌!@aE????@?i??L?????Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1-???g!@9-???g!@A-???g!@I-???g!@a???w?@?i7}?%*????Unknown
?AHostMul"Amean_absolute_percentage_error/mean_absolute_percentage_error/mul(1ffffff!@9ffffff!@Affffff!@Iffffff!@avU?됇@?i}%
L????Unknown
?BHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?????!@9?????!@A?????!@I?????!@aڧ"??>@?i??[????Unknown
tCHostAssignAddVariableOp"AssignAddVariableOp(1???S?? @9???S?? @A???S?? @I???S?? @a?
?{@?i`H?_????Unknown
uDHostReadVariableOp"div_no_nan/ReadVariableOp(1\???(? @9\???(? @A\???(? @I\???(? @a\???=@?i?>p`????Unknown
XEHostMean"Mean_1(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a??_F?=?is
??????Unknown
nFHostMul"Adam/ExponentialDecay/truediv(1??"??~@9??"??~@A??"??~@I??"??~@a??d?;<?i?6I?????Unknown
bGHostDivNoNan"div_no_nan_1(1+?Y@9+?Y@A+?Y@I+?Y@a??P?=?;?i)A?p????Unknown
?HHostAbs"Amean_absolute_percentage_error/mean_absolute_percentage_error/Abs(1??n?@@9??n?@@A??n?@@I??n?@@a?i1???:?iV?cGj????Unknown
~IHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?(\???@9?(\???@A?(\???@I?(\???@a???`??:?i	?o=?????Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_6(1     ?@9     ?@A     ?@I     ?@a????:?i??7 ????Unknown
?KHostDivNoNan"jgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/value/div_no_nan(1?(\??u@9?(\??u@A?(\??u@I?(\??u@a?I9??/8?i"<?.????Unknown
[LHostPow"
Adam/Pow_1(11?Zd@91?Zd@A1?Zd@I1?Zd@aY?=+8?i??
????Unknown
?MHostMaximum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Maximum(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a?E654?i??b??????Unknown
TNHostAbs"Abs(1??????@9??????@A??????@I??????@a???l?3?i???	????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_8(1?rh??|@9?rh??|@A?rh??|@I?rh??|@a}?(?v3?i??w?w????Unknown
oPHostReadVariableOp"Adam/ReadVariableOp(1?v???@9?v???@A?v???@I?v???@a?t?2?i1b:?????Unknown
?QHostBroadcastTo"Wgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/BroadcastTo(1y?&1?@9y?&1?@Ay?&1?@Iy?&1?@a?V?2?i?[=$????Unknown
VRHostCast"Cast(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@a?n7ŏ?1?i??T?X????Unknown
bSHostDivNoNan"div_no_nan_2(1333333@9333333@A333333@I333333@a?y??J1?iȲ23?????Unknown
bTHostDivNoNan"div_no_nan_3(1V-?@9V-?@AV-?@IV-?@a??'&??0?i?w?$?????Unknown
jUHostPow"Adam/ExponentialDecay/Pow(1?????M@9?????M@A?????M@I?????M@aw?38p0?iV??+?????Unknown
?VHostFloorDiv"Tgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/floordiv(1q=
ף?@9q=
ף?@Aq=
ף?@Iq=
ף?@a1:?u?0?i}?,"?????Unknown
?WHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1+????N@9+????N@A1?Z?@I1?Z?@a??{0?i??????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_3(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a? ??}.?i??{?????Unknown
wYHostReadVariableOp"div_no_nan_3/ReadVariableOp(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a???mP.?i??? x????Unknown
vZHostAssignAddVariableOp"AssignAddVariableOp_7(1?&1?@9?&1?@A?&1?@I?&1?@a??$<?-?i	PQ????Unknown
?[HostDivNoNan"Qmean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/value(1V-?@9V-?@AV-?@IV-?@a???p?5,?i4'Gb????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1B`??"?@9B`??"?@AB`??"?@IB`??"?@a?Fj?ev*?iح?Ȼ????Unknown
]]HostCast"Adam/Cast_1(1?l????
@9?l????
@A?l????
@I?l????
@arﶻ??)?iGiLRU????Unknown
T^HostMul"Mul(1L7?A`?	@9L7?A`?	@AL7?A`?	@IL7?A`?	@a??ټ?(?i???????Unknown
?_HostRealDiv"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv(1㥛? ?	@9㥛? ?	@A㥛? ?	@I㥛? ?	@akfH'g(?iu??`e????Unknown
X`HostCast"Cast_1(1???K7?@9???K7?@A???K7?@I???K7?@ay?h??N'?i?VxP?????Unknown
vaHostAssignAddVariableOp"AssignAddVariableOp_5(1?t?V@9?t?V@A?t?V@I?t?V@ae?*['?i?*6L????Unknown
vbHostAssignAddVariableOp"AssignAddVariableOp_1(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@ahb??!2%?i2HX?????Unknown
XcHostCast"Cast_2(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a?i?f?"%?i
?^??????Unknown
ydHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1?Q???@9?Q???@A?Q???@I?Q???@aX?s???$?iE}-?>????Unknown
?eHostCast"]mean_absolute_percentage_error/mean_absolute_percentage_error/weighted_loss/num_elements/Cast(1??????@9??????@A??????@I??????@a????$?iR????????Unknown
?fHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@agI????#?ig???????Unknown
?gHostNeg"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/sub/Neg(1+????@9+????@A+????@I+????@a>?"?iX?f>?????Unknown
?hHostSum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/mul/Sum(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a/y????"?i00`i????Unknown
?iHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1?"??~j@9?"??~j@A?"??~j@I?"??~j@a ??~!?i?@?S5????Unknown
yjHostReadVariableOp"div_no_nan_3/ReadVariableOp_1(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a???P?!?i?N>?F????Unknown
?kHostRealDiv"[gradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv/RealDiv(1?G?z@9?G?z@A?G?z@I?G?z@a$z???9 ?iǈ&IJ????Unknown
?lHostMul"Ugradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Abs_1/mul(1`??"?? @9`??"?? @A`??"?? @I`??"?? @a&I?s  ?iY=bPL????Unknown
ymHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?E??????9?E??????A?E??????I?E??????aI???]}?i/]Q;????Unknown
?nHostSum"Wgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/truediv/Sum(11?Zd??91?Zd??A1?Zd??I1?Zd??aY?=+?iϩ4?????Unknown
?oHostMean"Bmean_absolute_percentage_error/mean_absolute_percentage_error/Mean(15^?I??95^?I??A5^?I??I5^?I??a^?*???i?#?B?????Unknown
fpHostMul"Adam/ExponentialDecay(1!?rh????9!?rh????A!?rh????I!?rh????a?{????iJDo?E????Unknown
?qHostCast"Pgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/Cast(1?S㥛???9?S㥛???A?S㥛???I?S㥛???a#?O??iJé??????Unknown
arHostIdentity"Identity(1bX9????9bX9????AbX9????IbX9????a??U%??i;r?K?????Unknown?
?sHostSum"Sgradient_tape/mean_absolute_percentage_error/mean_absolute_percentage_error/sub/Sum(1y?&1???9y?&1???Ay?&1???Iy?&1???a?p?
??i?????????Unknown2CPU