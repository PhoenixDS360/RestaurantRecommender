
Æ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758û¤
¢
#Emb_month_vendor_created/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Emb_month_vendor_created/embeddings

7Emb_month_vendor_created/embeddings/Read/ReadVariableOpReadVariableOp#Emb_month_vendor_created/embeddings*
_output_shapes

:*
dtype0

Emb_Customer_City/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*-
shared_nameEmb_Customer_City/embeddings

0Emb_Customer_City/embeddings/Read/ReadVariableOpReadVariableOpEmb_Customer_City/embeddings*
_output_shapes

:7*
dtype0

Emb_Customer_Country/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Emb_Customer_Country/embeddings

3Emb_Customer_Country/embeddings/Read/ReadVariableOpReadVariableOpEmb_Customer_Country/embeddings*
_output_shapes

:*
dtype0

Emb_PrimaryTag/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-**
shared_nameEmb_PrimaryTag/embeddings

-Emb_PrimaryTag/embeddings/Read/ReadVariableOpReadVariableOpEmb_PrimaryTag/embeddings*
_output_shapes

:-*
dtype0

Emb_Vendor_Tags/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:G*+
shared_nameEmb_Vendor_Tags/embeddings

.Emb_Vendor_Tags/embeddings/Read/ReadVariableOpReadVariableOpEmb_Vendor_Tags/embeddings*
_output_shapes

:G*
dtype0

dense_layer_for_numbers/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z<*/
shared_name dense_layer_for_numbers/kernel

2dense_layer_for_numbers/kernel/Read/ReadVariableOpReadVariableOpdense_layer_for_numbers/kernel*
_output_shapes

:Z<*
dtype0

dense_layer_for_numbers/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_namedense_layer_for_numbers/bias

0dense_layer_for_numbers/bias/Read/ReadVariableOpReadVariableOpdense_layer_for_numbers/bias*
_output_shapes
:<*
dtype0

 dense_layer1_after_concat/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" dense_layer1_after_concat/kernel

4dense_layer1_after_concat/kernel/Read/ReadVariableOpReadVariableOp dense_layer1_after_concat/kernel*
_output_shapes

:d*
dtype0

dense_layer1_after_concat/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name dense_layer1_after_concat/bias

2dense_layer1_after_concat/bias/Read/ReadVariableOpReadVariableOpdense_layer1_after_concat/bias*
_output_shapes
:*
dtype0

 dense_layer2_after_concat/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" dense_layer2_after_concat/kernel

4dense_layer2_after_concat/kernel/Read/ReadVariableOpReadVariableOp dense_layer2_after_concat/kernel*
_output_shapes

:*
dtype0

dense_layer2_after_concat/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name dense_layer2_after_concat/bias

2dense_layer2_after_concat/bias/Read/ReadVariableOpReadVariableOpdense_layer2_after_concat/bias*
_output_shapes
:*
dtype0

 dense_layer3_after_concat/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" dense_layer3_after_concat/kernel

4dense_layer3_after_concat/kernel/Read/ReadVariableOpReadVariableOp dense_layer3_after_concat/kernel*
_output_shapes

:*
dtype0

dense_layer3_after_concat/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name dense_layer3_after_concat/bias

2dense_layer3_after_concat/bias/Read/ReadVariableOpReadVariableOpdense_layer3_after_concat/bias*
_output_shapes
:*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:*
dtype0
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
t
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nametrue_positives_2
m
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
: *
dtype0
v
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namefalse_positives_1
o
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
: *
dtype0
v
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namefalse_negatives_1
o
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
: *
dtype0
|
weights_intermediateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameweights_intermediate
u
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
: *
dtype0
x
true_positives_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_3
q
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes
:*
dtype0
z
false_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_2
s
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes
:*
dtype0
z
false_negatives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_2
s
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes
:*
dtype0

weights_intermediate_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameweights_intermediate_1
}
*weights_intermediate_1/Read/ReadVariableOpReadVariableOpweights_intermediate_1*
_output_shapes
:*
dtype0
°
*Adam/Emb_month_vendor_created/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/Emb_month_vendor_created/embeddings/m
©
>Adam/Emb_month_vendor_created/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/Emb_month_vendor_created/embeddings/m*
_output_shapes

:*
dtype0
¢
#Adam/Emb_Customer_City/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*4
shared_name%#Adam/Emb_Customer_City/embeddings/m

7Adam/Emb_Customer_City/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/Emb_Customer_City/embeddings/m*
_output_shapes

:7*
dtype0
¨
&Adam/Emb_Customer_Country/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/Emb_Customer_Country/embeddings/m
¡
:Adam/Emb_Customer_Country/embeddings/m/Read/ReadVariableOpReadVariableOp&Adam/Emb_Customer_Country/embeddings/m*
_output_shapes

:*
dtype0

 Adam/Emb_PrimaryTag/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*1
shared_name" Adam/Emb_PrimaryTag/embeddings/m

4Adam/Emb_PrimaryTag/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/Emb_PrimaryTag/embeddings/m*
_output_shapes

:-*
dtype0

!Adam/Emb_Vendor_Tags/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:G*2
shared_name#!Adam/Emb_Vendor_Tags/embeddings/m

5Adam/Emb_Vendor_Tags/embeddings/m/Read/ReadVariableOpReadVariableOp!Adam/Emb_Vendor_Tags/embeddings/m*
_output_shapes

:G*
dtype0
¦
%Adam/dense_layer_for_numbers/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z<*6
shared_name'%Adam/dense_layer_for_numbers/kernel/m

9Adam/dense_layer_for_numbers/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/dense_layer_for_numbers/kernel/m*
_output_shapes

:Z<*
dtype0

#Adam/dense_layer_for_numbers/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*4
shared_name%#Adam/dense_layer_for_numbers/bias/m

7Adam/dense_layer_for_numbers/bias/m/Read/ReadVariableOpReadVariableOp#Adam/dense_layer_for_numbers/bias/m*
_output_shapes
:<*
dtype0
ª
'Adam/dense_layer1_after_concat/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*8
shared_name)'Adam/dense_layer1_after_concat/kernel/m
£
;Adam/dense_layer1_after_concat/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/dense_layer1_after_concat/kernel/m*
_output_shapes

:d*
dtype0
¢
%Adam/dense_layer1_after_concat/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer1_after_concat/bias/m

9Adam/dense_layer1_after_concat/bias/m/Read/ReadVariableOpReadVariableOp%Adam/dense_layer1_after_concat/bias/m*
_output_shapes
:*
dtype0
ª
'Adam/dense_layer2_after_concat/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/dense_layer2_after_concat/kernel/m
£
;Adam/dense_layer2_after_concat/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/dense_layer2_after_concat/kernel/m*
_output_shapes

:*
dtype0
¢
%Adam/dense_layer2_after_concat/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer2_after_concat/bias/m

9Adam/dense_layer2_after_concat/bias/m/Read/ReadVariableOpReadVariableOp%Adam/dense_layer2_after_concat/bias/m*
_output_shapes
:*
dtype0
ª
'Adam/dense_layer3_after_concat/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/dense_layer3_after_concat/kernel/m
£
;Adam/dense_layer3_after_concat/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/dense_layer3_after_concat/kernel/m*
_output_shapes

:*
dtype0
¢
%Adam/dense_layer3_after_concat/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer3_after_concat/bias/m

9Adam/dense_layer3_after_concat/bias/m/Read/ReadVariableOpReadVariableOp%Adam/dense_layer3_after_concat/bias/m*
_output_shapes
:*
dtype0

Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Output/kernel/m
}
(Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/m*
_output_shapes

:*
dtype0
|
Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/m
u
&Adam/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output/bias/m*
_output_shapes
:*
dtype0
°
*Adam/Emb_month_vendor_created/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/Emb_month_vendor_created/embeddings/v
©
>Adam/Emb_month_vendor_created/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/Emb_month_vendor_created/embeddings/v*
_output_shapes

:*
dtype0
¢
#Adam/Emb_Customer_City/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*4
shared_name%#Adam/Emb_Customer_City/embeddings/v

7Adam/Emb_Customer_City/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/Emb_Customer_City/embeddings/v*
_output_shapes

:7*
dtype0
¨
&Adam/Emb_Customer_Country/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/Emb_Customer_Country/embeddings/v
¡
:Adam/Emb_Customer_Country/embeddings/v/Read/ReadVariableOpReadVariableOp&Adam/Emb_Customer_Country/embeddings/v*
_output_shapes

:*
dtype0

 Adam/Emb_PrimaryTag/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*1
shared_name" Adam/Emb_PrimaryTag/embeddings/v

4Adam/Emb_PrimaryTag/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/Emb_PrimaryTag/embeddings/v*
_output_shapes

:-*
dtype0

!Adam/Emb_Vendor_Tags/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:G*2
shared_name#!Adam/Emb_Vendor_Tags/embeddings/v

5Adam/Emb_Vendor_Tags/embeddings/v/Read/ReadVariableOpReadVariableOp!Adam/Emb_Vendor_Tags/embeddings/v*
_output_shapes

:G*
dtype0
¦
%Adam/dense_layer_for_numbers/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z<*6
shared_name'%Adam/dense_layer_for_numbers/kernel/v

9Adam/dense_layer_for_numbers/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/dense_layer_for_numbers/kernel/v*
_output_shapes

:Z<*
dtype0

#Adam/dense_layer_for_numbers/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*4
shared_name%#Adam/dense_layer_for_numbers/bias/v

7Adam/dense_layer_for_numbers/bias/v/Read/ReadVariableOpReadVariableOp#Adam/dense_layer_for_numbers/bias/v*
_output_shapes
:<*
dtype0
ª
'Adam/dense_layer1_after_concat/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*8
shared_name)'Adam/dense_layer1_after_concat/kernel/v
£
;Adam/dense_layer1_after_concat/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/dense_layer1_after_concat/kernel/v*
_output_shapes

:d*
dtype0
¢
%Adam/dense_layer1_after_concat/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer1_after_concat/bias/v

9Adam/dense_layer1_after_concat/bias/v/Read/ReadVariableOpReadVariableOp%Adam/dense_layer1_after_concat/bias/v*
_output_shapes
:*
dtype0
ª
'Adam/dense_layer2_after_concat/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/dense_layer2_after_concat/kernel/v
£
;Adam/dense_layer2_after_concat/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/dense_layer2_after_concat/kernel/v*
_output_shapes

:*
dtype0
¢
%Adam/dense_layer2_after_concat/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer2_after_concat/bias/v

9Adam/dense_layer2_after_concat/bias/v/Read/ReadVariableOpReadVariableOp%Adam/dense_layer2_after_concat/bias/v*
_output_shapes
:*
dtype0
ª
'Adam/dense_layer3_after_concat/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/dense_layer3_after_concat/kernel/v
£
;Adam/dense_layer3_after_concat/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/dense_layer3_after_concat/kernel/v*
_output_shapes

:*
dtype0
¢
%Adam/dense_layer3_after_concat/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_layer3_after_concat/bias/v

9Adam/dense_layer3_after_concat/bias/v/Read/ReadVariableOpReadVariableOp%Adam/dense_layer3_after_concat/bias/v*
_output_shapes
:*
dtype0

Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Output/kernel/v
}
(Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/v*
_output_shapes

:*
dtype0
|
Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/v
u
&Adam/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¹
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó
valueèBä BÜ

layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer_with_weights-7
layer-20
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!
signatures*
* 
* 
* 
* 
* 
 
"
embeddings
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
 
)
embeddings
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
 
0
embeddings
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
 
7
embeddings
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
 
>
embeddings
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
* 

E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 

K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 

Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 

]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
¦

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*

k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 
¦

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses*
¥
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}_random_generator
~__call__
*&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	iter
 beta_1
¡beta_2

¢decay
£learning_rate"m¥)m¦0m§7m¨>m©cmªdm«qm¬rm­	m®	m¯	m°	m±	m²	m³"v´)vµ0v¶7v·>v¸cv¹dvºqv»rv¼	v½	v¾	v¿	vÀ	vÁ	vÂ*
x
"0
)1
02
73
>4
c5
d6
q7
r8
9
10
11
12
13
14*
x
"0
)1
02
73
>4
c5
d6
q7
r8
9
10
11
12
13
14*
* 
µ
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

©serving_default* 
wq
VARIABLE_VALUE#Emb_month_vendor_created/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

"0*

"0*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
pj
VARIABLE_VALUEEmb_Customer_City/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

)0*

)0*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
sm
VARIABLE_VALUEEmb_Customer_Country/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

00*

00*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
mg
VARIABLE_VALUEEmb_PrimaryTag/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

70*

70*
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
nh
VARIABLE_VALUEEmb_Vendor_Tags/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

>0*

>0*
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
nh
VARIABLE_VALUEdense_layer_for_numbers/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEdense_layer_for_numbers/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

c0
d1*
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 
* 
* 
pj
VARIABLE_VALUE dense_layer1_after_concat/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEdense_layer1_after_concat/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

q0
r1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
pj
VARIABLE_VALUE dense_layer2_after_concat/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEdense_layer2_after_concat/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
pj
VARIABLE_VALUE dense_layer3_after_concat/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEdense_layer3_after_concat/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
º
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23*
,
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
`

thresholds
true_positives
false_negatives
	variables
	keras_api*
`

thresholds
true_positives
false_positives
	variables
	keras_api*


init_shape
true_positives
false_positives
false_negatives
weights_intermediate
	variables
	keras_api*


init_shape
true_positives
 false_positives
¡false_negatives
¢weights_intermediate
£	variables
¤	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
* 
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/3/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

	variables*
* 
ga
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEweights_intermediate_1Ckeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
$
0
 1
¡2
¢3*

£	variables*

VARIABLE_VALUE*Adam/Emb_month_vendor_created/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/Emb_Customer_City/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&Adam/Emb_Customer_Country/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/Emb_PrimaryTag/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Emb_Vendor_Tags/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer_for_numbers/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/dense_layer_for_numbers/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer1_after_concat/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer1_after_concat/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer2_after_concat/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer2_after_concat/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer3_after_concat/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer3_after_concat/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/Emb_month_vendor_created/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/Emb_Customer_City/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&Adam/Emb_Customer_Country/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/Emb_PrimaryTag/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/Emb_Vendor_Tags/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer_for_numbers/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/dense_layer_for_numbers/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer1_after_concat/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer1_after_concat/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer2_after_concat/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer2_after_concat/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'Adam/dense_layer3_after_concat/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_layer3_after_concat/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_customer_cityPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

 serving_default_customer_countryPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

$serving_default_month_vendor_createdPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

 serving_default_numerical_inputsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿZ
~
serving_default_primary_tagPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
~
serving_default_vendor_tagsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

»
StatefulPartitionedCallStatefulPartitionedCallserving_default_customer_city serving_default_customer_country$serving_default_month_vendor_created serving_default_numerical_inputsserving_default_primary_tagserving_default_vendor_tagsEmb_Vendor_Tags/embeddingsEmb_PrimaryTag/embeddingsEmb_Customer_Country/embeddingsEmb_Customer_City/embeddings#Emb_month_vendor_created/embeddingsdense_layer_for_numbers/kerneldense_layer_for_numbers/bias dense_layer1_after_concat/kerneldense_layer1_after_concat/bias dense_layer2_after_concat/kerneldense_layer2_after_concat/bias dense_layer3_after_concat/kerneldense_layer3_after_concat/biasOutput/kernelOutput/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3550895
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7Emb_month_vendor_created/embeddings/Read/ReadVariableOp0Emb_Customer_City/embeddings/Read/ReadVariableOp3Emb_Customer_Country/embeddings/Read/ReadVariableOp-Emb_PrimaryTag/embeddings/Read/ReadVariableOp.Emb_Vendor_Tags/embeddings/Read/ReadVariableOp2dense_layer_for_numbers/kernel/Read/ReadVariableOp0dense_layer_for_numbers/bias/Read/ReadVariableOp4dense_layer1_after_concat/kernel/Read/ReadVariableOp2dense_layer1_after_concat/bias/Read/ReadVariableOp4dense_layer2_after_concat/kernel/Read/ReadVariableOp2dense_layer2_after_concat/bias/Read/ReadVariableOp4dense_layer3_after_concat/kernel/Read/ReadVariableOp2dense_layer3_after_concat/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp$true_positives_3/Read/ReadVariableOp%false_positives_2/Read/ReadVariableOp%false_negatives_2/Read/ReadVariableOp*weights_intermediate_1/Read/ReadVariableOp>Adam/Emb_month_vendor_created/embeddings/m/Read/ReadVariableOp7Adam/Emb_Customer_City/embeddings/m/Read/ReadVariableOp:Adam/Emb_Customer_Country/embeddings/m/Read/ReadVariableOp4Adam/Emb_PrimaryTag/embeddings/m/Read/ReadVariableOp5Adam/Emb_Vendor_Tags/embeddings/m/Read/ReadVariableOp9Adam/dense_layer_for_numbers/kernel/m/Read/ReadVariableOp7Adam/dense_layer_for_numbers/bias/m/Read/ReadVariableOp;Adam/dense_layer1_after_concat/kernel/m/Read/ReadVariableOp9Adam/dense_layer1_after_concat/bias/m/Read/ReadVariableOp;Adam/dense_layer2_after_concat/kernel/m/Read/ReadVariableOp9Adam/dense_layer2_after_concat/bias/m/Read/ReadVariableOp;Adam/dense_layer3_after_concat/kernel/m/Read/ReadVariableOp9Adam/dense_layer3_after_concat/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp>Adam/Emb_month_vendor_created/embeddings/v/Read/ReadVariableOp7Adam/Emb_Customer_City/embeddings/v/Read/ReadVariableOp:Adam/Emb_Customer_Country/embeddings/v/Read/ReadVariableOp4Adam/Emb_PrimaryTag/embeddings/v/Read/ReadVariableOp5Adam/Emb_Vendor_Tags/embeddings/v/Read/ReadVariableOp9Adam/dense_layer_for_numbers/kernel/v/Read/ReadVariableOp7Adam/dense_layer_for_numbers/bias/v/Read/ReadVariableOp;Adam/dense_layer1_after_concat/kernel/v/Read/ReadVariableOp9Adam/dense_layer1_after_concat/bias/v/Read/ReadVariableOp;Adam/dense_layer2_after_concat/kernel/v/Read/ReadVariableOp9Adam/dense_layer2_after_concat/bias/v/Read/ReadVariableOp;Adam/dense_layer3_after_concat/kernel/v/Read/ReadVariableOp9Adam/dense_layer3_after_concat/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst*M
TinF
D2B	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3551414

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#Emb_month_vendor_created/embeddingsEmb_Customer_City/embeddingsEmb_Customer_Country/embeddingsEmb_PrimaryTag/embeddingsEmb_Vendor_Tags/embeddingsdense_layer_for_numbers/kerneldense_layer_for_numbers/bias dense_layer1_after_concat/kerneldense_layer1_after_concat/bias dense_layer2_after_concat/kerneldense_layer2_after_concat/bias dense_layer3_after_concat/kerneldense_layer3_after_concat/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_negativestrue_positives_1false_positivestrue_positives_2false_positives_1false_negatives_1weights_intermediatetrue_positives_3false_positives_2false_negatives_2weights_intermediate_1*Adam/Emb_month_vendor_created/embeddings/m#Adam/Emb_Customer_City/embeddings/m&Adam/Emb_Customer_Country/embeddings/m Adam/Emb_PrimaryTag/embeddings/m!Adam/Emb_Vendor_Tags/embeddings/m%Adam/dense_layer_for_numbers/kernel/m#Adam/dense_layer_for_numbers/bias/m'Adam/dense_layer1_after_concat/kernel/m%Adam/dense_layer1_after_concat/bias/m'Adam/dense_layer2_after_concat/kernel/m%Adam/dense_layer2_after_concat/bias/m'Adam/dense_layer3_after_concat/kernel/m%Adam/dense_layer3_after_concat/bias/mAdam/Output/kernel/mAdam/Output/bias/m*Adam/Emb_month_vendor_created/embeddings/v#Adam/Emb_Customer_City/embeddings/v&Adam/Emb_Customer_Country/embeddings/v Adam/Emb_PrimaryTag/embeddings/v!Adam/Emb_Vendor_Tags/embeddings/v%Adam/dense_layer_for_numbers/kernel/v#Adam/dense_layer_for_numbers/bias/v'Adam/dense_layer1_after_concat/kernel/v%Adam/dense_layer1_after_concat/bias/v'Adam/dense_layer2_after_concat/kernel/v%Adam/dense_layer2_after_concat/bias/v'Adam/dense_layer3_after_concat/kernel/v%Adam/dense_layer3_after_concat/bias/vAdam/Output/kernel/vAdam/Output/bias/v*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3551616ËÀ

F
*__inference_dropout1_layer_call_fn_3551106

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550198`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
C__inference_Output_layer_call_and_return_conditional_losses_3551194

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú|
ë
"__inference__wrapped_model_3549868
month_vendor_created
customer_city
customer_country
primary_tag
vendor_tags
numerical_inputs@
.model_emb_vendor_tags_embedding_lookup_3549789:G?
-model_emb_primarytag_embedding_lookup_3549795:-E
3model_emb_customer_country_embedding_lookup_3549801:B
0model_emb_customer_city_embedding_lookup_3549807:7I
7model_emb_month_vendor_created_embedding_lookup_3549813:N
<model_dense_layer_for_numbers_matmul_readvariableop_resource:Z<K
=model_dense_layer_for_numbers_biasadd_readvariableop_resource:<P
>model_dense_layer1_after_concat_matmul_readvariableop_resource:dM
?model_dense_layer1_after_concat_biasadd_readvariableop_resource:P
>model_dense_layer2_after_concat_matmul_readvariableop_resource:M
?model_dense_layer2_after_concat_biasadd_readvariableop_resource:P
>model_dense_layer3_after_concat_matmul_readvariableop_resource:M
?model_dense_layer3_after_concat_biasadd_readvariableop_resource:=
+model_output_matmul_readvariableop_resource::
,model_output_biasadd_readvariableop_resource:
identity¢(model/Emb_Customer_City/embedding_lookup¢+model/Emb_Customer_Country/embedding_lookup¢%model/Emb_PrimaryTag/embedding_lookup¢&model/Emb_Vendor_Tags/embedding_lookup¢/model/Emb_month_vendor_created/embedding_lookup¢#model/Output/BiasAdd/ReadVariableOp¢"model/Output/MatMul/ReadVariableOp¢6model/dense_layer1_after_concat/BiasAdd/ReadVariableOp¢5model/dense_layer1_after_concat/MatMul/ReadVariableOp¢6model/dense_layer2_after_concat/BiasAdd/ReadVariableOp¢5model/dense_layer2_after_concat/MatMul/ReadVariableOp¢6model/dense_layer3_after_concat/BiasAdd/ReadVariableOp¢5model/dense_layer3_after_concat/MatMul/ReadVariableOp¢4model/dense_layer_for_numbers/BiasAdd/ReadVariableOp¢3model/dense_layer_for_numbers/MatMul/ReadVariableOpp
model/Emb_Vendor_Tags/CastCastvendor_tags*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&model/Emb_Vendor_Tags/embedding_lookupResourceGather.model_emb_vendor_tags_embedding_lookup_3549789model/Emb_Vendor_Tags/Cast:y:0*
Tindices0*A
_class7
53loc:@model/Emb_Vendor_Tags/embedding_lookup/3549789*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0å
/model/Emb_Vendor_Tags/embedding_lookup/IdentityIdentity/model/Emb_Vendor_Tags/embedding_lookup:output:0*
T0*A
_class7
53loc:@model/Emb_Vendor_Tags/embedding_lookup/3549789*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
1model/Emb_Vendor_Tags/embedding_lookup/Identity_1Identity8model/Emb_Vendor_Tags/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
model/Emb_PrimaryTag/CastCastprimary_tag*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model/Emb_PrimaryTag/embedding_lookupResourceGather-model_emb_primarytag_embedding_lookup_3549795model/Emb_PrimaryTag/Cast:y:0*
Tindices0*@
_class6
42loc:@model/Emb_PrimaryTag/embedding_lookup/3549795*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0â
.model/Emb_PrimaryTag/embedding_lookup/IdentityIdentity.model/Emb_PrimaryTag/embedding_lookup:output:0*
T0*@
_class6
42loc:@model/Emb_PrimaryTag/embedding_lookup/3549795*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
0model/Emb_PrimaryTag/embedding_lookup/Identity_1Identity7model/Emb_PrimaryTag/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model/Emb_Customer_Country/CastCastcustomer_country*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
+model/Emb_Customer_Country/embedding_lookupResourceGather3model_emb_customer_country_embedding_lookup_3549801#model/Emb_Customer_Country/Cast:y:0*
Tindices0*F
_class<
:8loc:@model/Emb_Customer_Country/embedding_lookup/3549801*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0ô
4model/Emb_Customer_Country/embedding_lookup/IdentityIdentity4model/Emb_Customer_Country/embedding_lookup:output:0*
T0*F
_class<
:8loc:@model/Emb_Customer_Country/embedding_lookup/3549801*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
6model/Emb_Customer_Country/embedding_lookup/Identity_1Identity=model/Emb_Customer_Country/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model/Emb_Customer_City/CastCastcustomer_city*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/Emb_Customer_City/embedding_lookupResourceGather0model_emb_customer_city_embedding_lookup_3549807 model/Emb_Customer_City/Cast:y:0*
Tindices0*C
_class9
75loc:@model/Emb_Customer_City/embedding_lookup/3549807*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0ë
1model/Emb_Customer_City/embedding_lookup/IdentityIdentity1model/Emb_Customer_City/embedding_lookup:output:0*
T0*C
_class9
75loc:@model/Emb_Customer_City/embedding_lookup/3549807*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
3model/Emb_Customer_City/embedding_lookup/Identity_1Identity:model/Emb_Customer_City/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/Emb_month_vendor_created/CastCastmonth_vendor_created*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
/model/Emb_month_vendor_created/embedding_lookupResourceGather7model_emb_month_vendor_created_embedding_lookup_3549813'model/Emb_month_vendor_created/Cast:y:0*
Tindices0*J
_class@
><loc:@model/Emb_month_vendor_created/embedding_lookup/3549813*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0
8model/Emb_month_vendor_created/embedding_lookup/IdentityIdentity8model/Emb_month_vendor_created/embedding_lookup:output:0*
T0*J
_class@
><loc:@model/Emb_month_vendor_created/embedding_lookup/3549813*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
:model/Emb_month_vendor_created/embedding_lookup/Identity_1IdentityAmodel/Emb_month_vendor_created/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   µ
model/flatten/ReshapeReshapeCmodel/Emb_month_vendor_created/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ²
model/flatten_1/ReshapeReshape<model/Emb_Customer_City/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   µ
model/flatten_2/ReshapeReshape?model/Emb_Customer_Country/embedding_lookup/Identity_1:output:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¯
model/flatten_3/ReshapeReshape9model/Emb_PrimaryTag/embedding_lookup/Identity_1:output:0model/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   °
model/flatten_4/ReshapeReshape:model/Emb_Vendor_Tags/embedding_lookup/Identity_1:output:0model/flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3model/dense_layer_for_numbers/MatMul/ReadVariableOpReadVariableOp<model_dense_layer_for_numbers_matmul_readvariableop_resource*
_output_shapes

:Z<*
dtype0¯
$model/dense_layer_for_numbers/MatMulMatMulnumerical_inputs;model/dense_layer_for_numbers/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<®
4model/dense_layer_for_numbers/BiasAdd/ReadVariableOpReadVariableOp=model_dense_layer_for_numbers_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0Ð
%model/dense_layer_for_numbers/BiasAddBiasAdd.model/dense_layer_for_numbers/MatMul:product:0<model/dense_layer_for_numbers/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
"model/dense_layer_for_numbers/ReluRelu.model/dense_layer_for_numbers/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :á
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0 model/flatten_2/Reshape:output:0 model/flatten_3/Reshape:output:0 model/flatten_4/Reshape:output:00model/dense_layer_for_numbers/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
5model/dense_layer1_after_concat/MatMul/ReadVariableOpReadVariableOp>model_dense_layer1_after_concat_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ä
&model/dense_layer1_after_concat/MatMulMatMul!model/concatenate/concat:output:0=model/dense_layer1_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
6model/dense_layer1_after_concat/BiasAdd/ReadVariableOpReadVariableOp?model_dense_layer1_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ö
'model/dense_layer1_after_concat/BiasAddBiasAdd0model/dense_layer1_after_concat/MatMul:product:0>model/dense_layer1_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_layer1_after_concat/ReluRelu0model/dense_layer1_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout1/IdentityIdentity2model/dense_layer1_after_concat/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
5model/dense_layer2_after_concat/MatMul/ReadVariableOpReadVariableOp>model_dense_layer2_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
&model/dense_layer2_after_concat/MatMulMatMul model/dropout1/Identity:output:0=model/dense_layer2_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
6model/dense_layer2_after_concat/BiasAdd/ReadVariableOpReadVariableOp?model_dense_layer2_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ö
'model/dense_layer2_after_concat/BiasAddBiasAdd0model/dense_layer2_after_concat/MatMul:product:0>model/dense_layer2_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_layer2_after_concat/ReluRelu0model/dense_layer2_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout2/IdentityIdentity2model/dense_layer2_after_concat/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
5model/dense_layer3_after_concat/MatMul/ReadVariableOpReadVariableOp>model_dense_layer3_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
&model/dense_layer3_after_concat/MatMulMatMul model/dropout2/Identity:output:0=model/dense_layer3_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
6model/dense_layer3_after_concat/BiasAdd/ReadVariableOpReadVariableOp?model_dense_layer3_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ö
'model/dense_layer3_after_concat/BiasAddBiasAdd0model/dense_layer3_after_concat/MatMul:product:0>model/dense_layer3_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_layer3_after_concat/ReluRelu0model/dense_layer3_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/Output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¯
model/Output/MatMulMatMul2model/dense_layer3_after_concat/Relu:activations:0*model/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/Output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/Output/BiasAddBiasAddmodel/Output/MatMul:product:0+model/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model/Output/SoftmaxSoftmaxmodel/Output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitymodel/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp)^model/Emb_Customer_City/embedding_lookup,^model/Emb_Customer_Country/embedding_lookup&^model/Emb_PrimaryTag/embedding_lookup'^model/Emb_Vendor_Tags/embedding_lookup0^model/Emb_month_vendor_created/embedding_lookup$^model/Output/BiasAdd/ReadVariableOp#^model/Output/MatMul/ReadVariableOp7^model/dense_layer1_after_concat/BiasAdd/ReadVariableOp6^model/dense_layer1_after_concat/MatMul/ReadVariableOp7^model/dense_layer2_after_concat/BiasAdd/ReadVariableOp6^model/dense_layer2_after_concat/MatMul/ReadVariableOp7^model/dense_layer3_after_concat/BiasAdd/ReadVariableOp6^model/dense_layer3_after_concat/MatMul/ReadVariableOp5^model/dense_layer_for_numbers/BiasAdd/ReadVariableOp4^model/dense_layer_for_numbers/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2T
(model/Emb_Customer_City/embedding_lookup(model/Emb_Customer_City/embedding_lookup2Z
+model/Emb_Customer_Country/embedding_lookup+model/Emb_Customer_Country/embedding_lookup2N
%model/Emb_PrimaryTag/embedding_lookup%model/Emb_PrimaryTag/embedding_lookup2P
&model/Emb_Vendor_Tags/embedding_lookup&model/Emb_Vendor_Tags/embedding_lookup2b
/model/Emb_month_vendor_created/embedding_lookup/model/Emb_month_vendor_created/embedding_lookup2J
#model/Output/BiasAdd/ReadVariableOp#model/Output/BiasAdd/ReadVariableOp2H
"model/Output/MatMul/ReadVariableOp"model/Output/MatMul/ReadVariableOp2p
6model/dense_layer1_after_concat/BiasAdd/ReadVariableOp6model/dense_layer1_after_concat/BiasAdd/ReadVariableOp2n
5model/dense_layer1_after_concat/MatMul/ReadVariableOp5model/dense_layer1_after_concat/MatMul/ReadVariableOp2p
6model/dense_layer2_after_concat/BiasAdd/ReadVariableOp6model/dense_layer2_after_concat/BiasAdd/ReadVariableOp2n
5model/dense_layer2_after_concat/MatMul/ReadVariableOp5model/dense_layer2_after_concat/MatMul/ReadVariableOp2p
6model/dense_layer3_after_concat/BiasAdd/ReadVariableOp6model/dense_layer3_after_concat/BiasAdd/ReadVariableOp2n
5model/dense_layer3_after_concat/MatMul/ReadVariableOp5model/dense_layer3_after_concat/MatMul/ReadVariableOp2l
4model/dense_layer_for_numbers/BiasAdd/ReadVariableOp4model/dense_layer_for_numbers/BiasAdd/ReadVariableOp2j
3model/dense_layer_for_numbers/MatMul/ReadVariableOp3model/dense_layer_for_numbers/MatMul/ReadVariableOp:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs
æ
¨
;__inference_dense_layer2_after_concat_layer_call_fn_3551124

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout2_layer_call_and_return_conditional_losses_3550071

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
G
+__inference_flatten_3_layer_call_fn_3551018

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«	
©
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909

inputs*
embedding_lookup_3549903:-
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3549903Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3549903*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3549903*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÎU
±

B__inference_model_layer_call_and_return_conditional_losses_3550406

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5)
emb_vendor_tags_3550357:G(
emb_primarytag_3550360:-.
emb_customer_country_3550363:+
emb_customer_city_3550366:72
 emb_month_vendor_created_3550369:1
dense_layer_for_numbers_3550377:Z<-
dense_layer_for_numbers_3550379:<3
!dense_layer1_after_concat_3550383:d/
!dense_layer1_after_concat_3550385:3
!dense_layer2_after_concat_3550389:/
!dense_layer2_after_concat_3550391:3
!dense_layer3_after_concat_3550395:/
!dense_layer3_after_concat_3550397: 
output_3550400:
output_3550402:
identity¢)Emb_Customer_City/StatefulPartitionedCall¢,Emb_Customer_Country/StatefulPartitionedCall¢&Emb_PrimaryTag/StatefulPartitionedCall¢'Emb_Vendor_Tags/StatefulPartitionedCall¢0Emb_month_vendor_created/StatefulPartitionedCall¢Output/StatefulPartitionedCall¢1dense_layer1_after_concat/StatefulPartitionedCall¢1dense_layer2_after_concat/StatefulPartitionedCall¢1dense_layer3_after_concat/StatefulPartitionedCall¢/dense_layer_for_numbers/StatefulPartitionedCallú
'Emb_Vendor_Tags/StatefulPartitionedCallStatefulPartitionedCallinputs_4emb_vendor_tags_3550357*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895÷
&Emb_PrimaryTag/StatefulPartitionedCallStatefulPartitionedCallinputs_3emb_primarytag_3550360*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909
,Emb_Customer_Country/StatefulPartitionedCallStatefulPartitionedCallinputs_2emb_customer_country_3550363*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923
)Emb_Customer_City/StatefulPartitionedCallStatefulPartitionedCallinputs_1emb_customer_city_3550366*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937
0Emb_month_vendor_created/StatefulPartitionedCallStatefulPartitionedCallinputs emb_month_vendor_created_3550369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951ê
flatten/PartitionedCallPartitionedCall9Emb_month_vendor_created/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3549961ç
flatten_1/PartitionedCallPartitionedCall2Emb_Customer_City/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969ê
flatten_2/PartitionedCallPartitionedCall5Emb_Customer_Country/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977ä
flatten_3/PartitionedCallPartitionedCall/Emb_PrimaryTag/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985å
flatten_4/PartitionedCallPartitionedCall0Emb_Vendor_Tags/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993±
/dense_layer_for_numbers/StatefulPartitionedCallStatefulPartitionedCallinputs_5dense_layer_for_numbers_3550377dense_layer_for_numbers_3550379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006¨
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:08dense_layer_for_numbers/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023Õ
1dense_layer1_after_concat/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0!dense_layer1_after_concat_3550383!dense_layer1_after_concat_3550385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036í
dropout1/PartitionedCallPartitionedCall:dense_layer1_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550198Ò
1dense_layer2_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0!dense_layer2_after_concat_3550389!dense_layer2_after_concat_3550391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060í
dropout2/PartitionedCallPartitionedCall:dense_layer2_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550173Ò
1dense_layer3_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0!dense_layer3_after_concat_3550395!dense_layer3_after_concat_3550397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084
Output/StatefulPartitionedCallStatefulPartitionedCall:dense_layer3_after_concat/StatefulPartitionedCall:output:0output_3550400output_3550402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_3550101v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp*^Emb_Customer_City/StatefulPartitionedCall-^Emb_Customer_Country/StatefulPartitionedCall'^Emb_PrimaryTag/StatefulPartitionedCall(^Emb_Vendor_Tags/StatefulPartitionedCall1^Emb_month_vendor_created/StatefulPartitionedCall^Output/StatefulPartitionedCall2^dense_layer1_after_concat/StatefulPartitionedCall2^dense_layer2_after_concat/StatefulPartitionedCall2^dense_layer3_after_concat/StatefulPartitionedCall0^dense_layer_for_numbers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2V
)Emb_Customer_City/StatefulPartitionedCall)Emb_Customer_City/StatefulPartitionedCall2\
,Emb_Customer_Country/StatefulPartitionedCall,Emb_Customer_Country/StatefulPartitionedCall2P
&Emb_PrimaryTag/StatefulPartitionedCall&Emb_PrimaryTag/StatefulPartitionedCall2R
'Emb_Vendor_Tags/StatefulPartitionedCall'Emb_Vendor_Tags/StatefulPartitionedCall2d
0Emb_month_vendor_created/StatefulPartitionedCall0Emb_month_vendor_created/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2f
1dense_layer1_after_concat/StatefulPartitionedCall1dense_layer1_after_concat/StatefulPartitionedCall2f
1dense_layer2_after_concat/StatefulPartitionedCall1dense_layer2_after_concat/StatefulPartitionedCall2f
1dense_layer3_after_concat/StatefulPartitionedCall1dense_layer3_after_concat/StatefulPartitionedCall2b
/dense_layer_for_numbers/StatefulPartitionedCall/dense_layer_for_numbers/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
±	
¯
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923

inputs*
embedding_lookup_3549917:
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3549917Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3549917*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3549917*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±	
¯
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3550946

inputs*
embedding_lookup_3550940:
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3550940Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3550940*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3550940*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_3551024

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_3549961

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
É
'__inference_model_layer_call_fn_3550639
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:G
	unknown_0:-
	unknown_1:
	unknown_2:7
	unknown_3:
	unknown_4:Z<
	unknown_5:<
	unknown_6:d
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3550108o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
"
_user_specified_name
inputs/5
Ø
c
E__inference_dropout2_layer_call_and_return_conditional_losses_3551150

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

6__inference_Emb_Customer_Country_layer_call_fn_3550936

inputs
unknown:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
å
 __inference__traced_save_3551414
file_prefixB
>savev2_emb_month_vendor_created_embeddings_read_readvariableop;
7savev2_emb_customer_city_embeddings_read_readvariableop>
:savev2_emb_customer_country_embeddings_read_readvariableop8
4savev2_emb_primarytag_embeddings_read_readvariableop9
5savev2_emb_vendor_tags_embeddings_read_readvariableop=
9savev2_dense_layer_for_numbers_kernel_read_readvariableop;
7savev2_dense_layer_for_numbers_bias_read_readvariableop?
;savev2_dense_layer1_after_concat_kernel_read_readvariableop=
9savev2_dense_layer1_after_concat_bias_read_readvariableop?
;savev2_dense_layer2_after_concat_kernel_read_readvariableop=
9savev2_dense_layer2_after_concat_bias_read_readvariableop?
;savev2_dense_layer3_after_concat_kernel_read_readvariableop=
9savev2_dense_layer3_after_concat_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop/
+savev2_true_positives_3_read_readvariableop0
,savev2_false_positives_2_read_readvariableop0
,savev2_false_negatives_2_read_readvariableop5
1savev2_weights_intermediate_1_read_readvariableopI
Esavev2_adam_emb_month_vendor_created_embeddings_m_read_readvariableopB
>savev2_adam_emb_customer_city_embeddings_m_read_readvariableopE
Asavev2_adam_emb_customer_country_embeddings_m_read_readvariableop?
;savev2_adam_emb_primarytag_embeddings_m_read_readvariableop@
<savev2_adam_emb_vendor_tags_embeddings_m_read_readvariableopD
@savev2_adam_dense_layer_for_numbers_kernel_m_read_readvariableopB
>savev2_adam_dense_layer_for_numbers_bias_m_read_readvariableopF
Bsavev2_adam_dense_layer1_after_concat_kernel_m_read_readvariableopD
@savev2_adam_dense_layer1_after_concat_bias_m_read_readvariableopF
Bsavev2_adam_dense_layer2_after_concat_kernel_m_read_readvariableopD
@savev2_adam_dense_layer2_after_concat_bias_m_read_readvariableopF
Bsavev2_adam_dense_layer3_after_concat_kernel_m_read_readvariableopD
@savev2_adam_dense_layer3_after_concat_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableopI
Esavev2_adam_emb_month_vendor_created_embeddings_v_read_readvariableopB
>savev2_adam_emb_customer_city_embeddings_v_read_readvariableopE
Asavev2_adam_emb_customer_country_embeddings_v_read_readvariableop?
;savev2_adam_emb_primarytag_embeddings_v_read_readvariableop@
<savev2_adam_emb_vendor_tags_embeddings_v_read_readvariableopD
@savev2_adam_dense_layer_for_numbers_kernel_v_read_readvariableopB
>savev2_adam_dense_layer_for_numbers_bias_v_read_readvariableopF
Bsavev2_adam_dense_layer1_after_concat_kernel_v_read_readvariableopD
@savev2_adam_dense_layer1_after_concat_bias_v_read_readvariableopF
Bsavev2_adam_dense_layer2_after_concat_kernel_v_read_readvariableopD
@savev2_adam_dense_layer2_after_concat_bias_v_read_readvariableopF
Bsavev2_adam_dense_layer3_after_concat_kernel_v_read_readvariableopD
@savev2_adam_dense_layer3_after_concat_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: $
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*È#
value¾#B»#AB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/3/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHò
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B è
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_emb_month_vendor_created_embeddings_read_readvariableop7savev2_emb_customer_city_embeddings_read_readvariableop:savev2_emb_customer_country_embeddings_read_readvariableop4savev2_emb_primarytag_embeddings_read_readvariableop5savev2_emb_vendor_tags_embeddings_read_readvariableop9savev2_dense_layer_for_numbers_kernel_read_readvariableop7savev2_dense_layer_for_numbers_bias_read_readvariableop;savev2_dense_layer1_after_concat_kernel_read_readvariableop9savev2_dense_layer1_after_concat_bias_read_readvariableop;savev2_dense_layer2_after_concat_kernel_read_readvariableop9savev2_dense_layer2_after_concat_bias_read_readvariableop;savev2_dense_layer3_after_concat_kernel_read_readvariableop9savev2_dense_layer3_after_concat_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop/savev2_weights_intermediate_read_readvariableop+savev2_true_positives_3_read_readvariableop,savev2_false_positives_2_read_readvariableop,savev2_false_negatives_2_read_readvariableop1savev2_weights_intermediate_1_read_readvariableopEsavev2_adam_emb_month_vendor_created_embeddings_m_read_readvariableop>savev2_adam_emb_customer_city_embeddings_m_read_readvariableopAsavev2_adam_emb_customer_country_embeddings_m_read_readvariableop;savev2_adam_emb_primarytag_embeddings_m_read_readvariableop<savev2_adam_emb_vendor_tags_embeddings_m_read_readvariableop@savev2_adam_dense_layer_for_numbers_kernel_m_read_readvariableop>savev2_adam_dense_layer_for_numbers_bias_m_read_readvariableopBsavev2_adam_dense_layer1_after_concat_kernel_m_read_readvariableop@savev2_adam_dense_layer1_after_concat_bias_m_read_readvariableopBsavev2_adam_dense_layer2_after_concat_kernel_m_read_readvariableop@savev2_adam_dense_layer2_after_concat_bias_m_read_readvariableopBsavev2_adam_dense_layer3_after_concat_kernel_m_read_readvariableop@savev2_adam_dense_layer3_after_concat_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableopEsavev2_adam_emb_month_vendor_created_embeddings_v_read_readvariableop>savev2_adam_emb_customer_city_embeddings_v_read_readvariableopAsavev2_adam_emb_customer_country_embeddings_v_read_readvariableop;savev2_adam_emb_primarytag_embeddings_v_read_readvariableop<savev2_adam_emb_vendor_tags_embeddings_v_read_readvariableop@savev2_adam_dense_layer_for_numbers_kernel_v_read_readvariableop>savev2_adam_dense_layer_for_numbers_bias_v_read_readvariableopBsavev2_adam_dense_layer1_after_concat_kernel_v_read_readvariableop@savev2_adam_dense_layer1_after_concat_bias_v_read_readvariableopBsavev2_adam_dense_layer2_after_concat_kernel_v_read_readvariableop@savev2_adam_dense_layer2_after_concat_bias_v_read_readvariableopBsavev2_adam_dense_layer3_after_concat_kernel_v_read_readvariableop@savev2_adam_dense_layer3_after_concat_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*å
_input_shapesÓ
Ð: ::7::-:G:Z<:<:d:::::::: : : : : : : ::::: : : : ::::::7::-:G:Z<:<:d:::::::::7::-:G:Z<:<:d:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

:7:$ 

_output_shapes

::$ 

_output_shapes

:-:$ 

_output_shapes

:G:$ 

_output_shapes

:Z<: 

_output_shapes
:<:$ 

_output_shapes

:d: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
::$# 

_output_shapes

::$$ 

_output_shapes

:7:$% 

_output_shapes

::$& 

_output_shapes

:-:$' 

_output_shapes

:G:$( 

_output_shapes

:Z<: )

_output_shapes
:<:$* 

_output_shapes

:d: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

::$3 

_output_shapes

:7:$4 

_output_shapes

::$5 

_output_shapes

:-:$6 

_output_shapes

:G:$7 

_output_shapes

:Z<: 8

_output_shapes
:<:$9 

_output_shapes

:d: :

_output_shapes
::$; 

_output_shapes

:: <

_output_shapes
::$= 

_output_shapes

:: >

_output_shapes
::$? 

_output_shapes

:: @

_output_shapes
::A

_output_shapes
: 
Ä

:__inference_Emb_month_vendor_created_layer_call_fn_3550902

inputs
unknown:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

(__inference_Output_layer_call_fn_3551183

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_3550101o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®	
¬
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937

inputs*
embedding_lookup_3549931:7
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3549931Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3549931*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3549931*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÓV
Ú

B__inference_model_layer_call_and_return_conditional_losses_3550536
month_vendor_created
customer_city
customer_country
primary_tag
vendor_tags
numerical_inputs)
emb_vendor_tags_3550487:G(
emb_primarytag_3550490:-.
emb_customer_country_3550493:+
emb_customer_city_3550496:72
 emb_month_vendor_created_3550499:1
dense_layer_for_numbers_3550507:Z<-
dense_layer_for_numbers_3550509:<3
!dense_layer1_after_concat_3550513:d/
!dense_layer1_after_concat_3550515:3
!dense_layer2_after_concat_3550519:/
!dense_layer2_after_concat_3550521:3
!dense_layer3_after_concat_3550525:/
!dense_layer3_after_concat_3550527: 
output_3550530:
output_3550532:
identity¢)Emb_Customer_City/StatefulPartitionedCall¢,Emb_Customer_Country/StatefulPartitionedCall¢&Emb_PrimaryTag/StatefulPartitionedCall¢'Emb_Vendor_Tags/StatefulPartitionedCall¢0Emb_month_vendor_created/StatefulPartitionedCall¢Output/StatefulPartitionedCall¢1dense_layer1_after_concat/StatefulPartitionedCall¢1dense_layer2_after_concat/StatefulPartitionedCall¢1dense_layer3_after_concat/StatefulPartitionedCall¢/dense_layer_for_numbers/StatefulPartitionedCallý
'Emb_Vendor_Tags/StatefulPartitionedCallStatefulPartitionedCallvendor_tagsemb_vendor_tags_3550487*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895ú
&Emb_PrimaryTag/StatefulPartitionedCallStatefulPartitionedCallprimary_tagemb_primarytag_3550490*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909
,Emb_Customer_Country/StatefulPartitionedCallStatefulPartitionedCallcustomer_countryemb_customer_country_3550493*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923
)Emb_Customer_City/StatefulPartitionedCallStatefulPartitionedCallcustomer_cityemb_customer_city_3550496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937¡
0Emb_month_vendor_created/StatefulPartitionedCallStatefulPartitionedCallmonth_vendor_created emb_month_vendor_created_3550499*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951ê
flatten/PartitionedCallPartitionedCall9Emb_month_vendor_created/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3549961ç
flatten_1/PartitionedCallPartitionedCall2Emb_Customer_City/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969ê
flatten_2/PartitionedCallPartitionedCall5Emb_Customer_Country/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977ä
flatten_3/PartitionedCallPartitionedCall/Emb_PrimaryTag/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985å
flatten_4/PartitionedCallPartitionedCall0Emb_Vendor_Tags/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993¹
/dense_layer_for_numbers/StatefulPartitionedCallStatefulPartitionedCallnumerical_inputsdense_layer_for_numbers_3550507dense_layer_for_numbers_3550509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006¨
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:08dense_layer_for_numbers/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023Õ
1dense_layer1_after_concat/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0!dense_layer1_after_concat_3550513!dense_layer1_after_concat_3550515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036í
dropout1/PartitionedCallPartitionedCall:dense_layer1_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550047Ò
1dense_layer2_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0!dense_layer2_after_concat_3550519!dense_layer2_after_concat_3550521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060í
dropout2/PartitionedCallPartitionedCall:dense_layer2_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550071Ò
1dense_layer3_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0!dense_layer3_after_concat_3550525!dense_layer3_after_concat_3550527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084
Output/StatefulPartitionedCallStatefulPartitionedCall:dense_layer3_after_concat/StatefulPartitionedCall:output:0output_3550530output_3550532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_3550101v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp*^Emb_Customer_City/StatefulPartitionedCall-^Emb_Customer_Country/StatefulPartitionedCall'^Emb_PrimaryTag/StatefulPartitionedCall(^Emb_Vendor_Tags/StatefulPartitionedCall1^Emb_month_vendor_created/StatefulPartitionedCall^Output/StatefulPartitionedCall2^dense_layer1_after_concat/StatefulPartitionedCall2^dense_layer2_after_concat/StatefulPartitionedCall2^dense_layer3_after_concat/StatefulPartitionedCall0^dense_layer_for_numbers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2V
)Emb_Customer_City/StatefulPartitionedCall)Emb_Customer_City/StatefulPartitionedCall2\
,Emb_Customer_Country/StatefulPartitionedCall,Emb_Customer_Country/StatefulPartitionedCall2P
&Emb_PrimaryTag/StatefulPartitionedCall&Emb_PrimaryTag/StatefulPartitionedCall2R
'Emb_Vendor_Tags/StatefulPartitionedCall'Emb_Vendor_Tags/StatefulPartitionedCall2d
0Emb_month_vendor_created/StatefulPartitionedCall0Emb_month_vendor_created/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2f
1dense_layer1_after_concat/StatefulPartitionedCall1dense_layer1_after_concat/StatefulPartitionedCall2f
1dense_layer2_after_concat/StatefulPartitionedCall1dense_layer2_after_concat/StatefulPartitionedCall2f
1dense_layer3_after_concat/StatefulPartitionedCall1dense_layer3_after_concat/StatefulPartitionedCall2b
/dense_layer_for_numbers/StatefulPartitionedCall/dense_layer_for_numbers/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs
Ø
c
E__inference_dropout1_layer_call_and_return_conditional_losses_3550047

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼	
¬
H__inference_concatenate_layer_call_and_return_conditional_losses_3551076
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ<:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
"
_user_specified_name
inputs/5
ë
ð
'__inference_model_layer_call_fn_3550479
month_vendor_created
customer_city
customer_country
primary_tag
vendor_tags
numerical_inputs
unknown:G
	unknown_0:-
	unknown_1:
	unknown_2:7
	unknown_3:
	unknown_4:Z<
	unknown_5:<
	unknown_6:d
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallmonth_vendor_createdcustomer_citycustomer_countryprimary_tagvendor_tagsnumerical_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3550406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs
­


V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ë
ð
'__inference_model_layer_call_fn_3550141
month_vendor_created
customer_city
customer_country
primary_tag
vendor_tags
numerical_inputs
unknown:G
	unknown_0:-
	unknown_1:
	unknown_2:7
	unknown_3:
	unknown_4:Z<
	unknown_5:<
	unknown_6:d
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallmonth_vendor_createdcustomer_citycustomer_countryprimary_tagvendor_tagsnumerical_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3550108o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs
­


V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
É
'__inference_model_layer_call_fn_3550679
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:G
	unknown_0:-
	unknown_1:
	unknown_2:7
	unknown_3:
	unknown_4:Z<
	unknown_5:<
	unknown_6:d
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3550406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
"
_user_specified_name
inputs/5
«	
©
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3550963

inputs*
embedding_lookup_3550957:-
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3550957Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3550957*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3550957*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
a
E__inference_dropout2_layer_call_and_return_conditional_losses_3551154

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
a
E__inference_dropout1_layer_call_and_return_conditional_losses_3551115

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
î
%__inference_signature_wrapper_3550895
customer_city
customer_country
month_vendor_created
numerical_inputs
primary_tag
vendor_tags
unknown:G
	unknown_0:-
	unknown_1:
	unknown_2:7
	unknown_3:
	unknown_4:Z<
	unknown_5:<
	unknown_6:d
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallmonth_vendor_createdcustomer_citycustomer_countryprimary_tagvendor_tagsnumerical_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_3549868o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿZ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags
¾
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_3551013

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_3551035

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
­


V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3551174

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬	
ª
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895

inputs*
embedding_lookup_3549889:G
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
embedding_lookupResourceGatherembedding_lookup_3549889Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3549889*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3549889*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
µ	
³
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3550912

inputs*
embedding_lookup_3550906:
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3550906Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3550906*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3550906*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æq
°
B__inference_model_layer_call_and_return_conditional_losses_3550767
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5:
(emb_vendor_tags_embedding_lookup_3550688:G9
'emb_primarytag_embedding_lookup_3550694:-?
-emb_customer_country_embedding_lookup_3550700:<
*emb_customer_city_embedding_lookup_3550706:7C
1emb_month_vendor_created_embedding_lookup_3550712:H
6dense_layer_for_numbers_matmul_readvariableop_resource:Z<E
7dense_layer_for_numbers_biasadd_readvariableop_resource:<J
8dense_layer1_after_concat_matmul_readvariableop_resource:dG
9dense_layer1_after_concat_biasadd_readvariableop_resource:J
8dense_layer2_after_concat_matmul_readvariableop_resource:G
9dense_layer2_after_concat_biasadd_readvariableop_resource:J
8dense_layer3_after_concat_matmul_readvariableop_resource:G
9dense_layer3_after_concat_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity¢"Emb_Customer_City/embedding_lookup¢%Emb_Customer_Country/embedding_lookup¢Emb_PrimaryTag/embedding_lookup¢ Emb_Vendor_Tags/embedding_lookup¢)Emb_month_vendor_created/embedding_lookup¢Output/BiasAdd/ReadVariableOp¢Output/MatMul/ReadVariableOp¢0dense_layer1_after_concat/BiasAdd/ReadVariableOp¢/dense_layer1_after_concat/MatMul/ReadVariableOp¢0dense_layer2_after_concat/BiasAdd/ReadVariableOp¢/dense_layer2_after_concat/MatMul/ReadVariableOp¢0dense_layer3_after_concat/BiasAdd/ReadVariableOp¢/dense_layer3_after_concat/MatMul/ReadVariableOp¢.dense_layer_for_numbers/BiasAdd/ReadVariableOp¢-dense_layer_for_numbers/MatMul/ReadVariableOpg
Emb_Vendor_Tags/CastCastinputs_4*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý
 Emb_Vendor_Tags/embedding_lookupResourceGather(emb_vendor_tags_embedding_lookup_3550688Emb_Vendor_Tags/Cast:y:0*
Tindices0*;
_class1
/-loc:@Emb_Vendor_Tags/embedding_lookup/3550688*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0Ó
)Emb_Vendor_Tags/embedding_lookup/IdentityIdentity)Emb_Vendor_Tags/embedding_lookup:output:0*
T0*;
_class1
/-loc:@Emb_Vendor_Tags/embedding_lookup/3550688*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
+Emb_Vendor_Tags/embedding_lookup/Identity_1Identity2Emb_Vendor_Tags/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
Emb_PrimaryTag/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
Emb_PrimaryTag/embedding_lookupResourceGather'emb_primarytag_embedding_lookup_3550694Emb_PrimaryTag/Cast:y:0*
Tindices0*:
_class0
.,loc:@Emb_PrimaryTag/embedding_lookup/3550694*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ð
(Emb_PrimaryTag/embedding_lookup/IdentityIdentity(Emb_PrimaryTag/embedding_lookup:output:0*
T0*:
_class0
.,loc:@Emb_PrimaryTag/embedding_lookup/3550694*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*Emb_PrimaryTag/embedding_lookup/Identity_1Identity1Emb_PrimaryTag/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Emb_Customer_Country/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Emb_Customer_Country/embedding_lookupResourceGather-emb_customer_country_embedding_lookup_3550700Emb_Customer_Country/Cast:y:0*
Tindices0*@
_class6
42loc:@Emb_Customer_Country/embedding_lookup/3550700*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0â
.Emb_Customer_Country/embedding_lookup/IdentityIdentity.Emb_Customer_Country/embedding_lookup:output:0*
T0*@
_class6
42loc:@Emb_Customer_Country/embedding_lookup/3550700*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
0Emb_Customer_Country/embedding_lookup/Identity_1Identity7Emb_Customer_Country/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
Emb_Customer_City/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Emb_Customer_City/embedding_lookupResourceGather*emb_customer_city_embedding_lookup_3550706Emb_Customer_City/Cast:y:0*
Tindices0*=
_class3
1/loc:@Emb_Customer_City/embedding_lookup/3550706*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ù
+Emb_Customer_City/embedding_lookup/IdentityIdentity+Emb_Customer_City/embedding_lookup:output:0*
T0*=
_class3
1/loc:@Emb_Customer_City/embedding_lookup/3550706*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-Emb_Customer_City/embedding_lookup/Identity_1Identity4Emb_Customer_City/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Emb_month_vendor_created/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
)Emb_month_vendor_created/embedding_lookupResourceGather1emb_month_vendor_created_embedding_lookup_3550712!Emb_month_vendor_created/Cast:y:0*
Tindices0*D
_class:
86loc:@Emb_month_vendor_created/embedding_lookup/3550712*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0î
2Emb_month_vendor_created/embedding_lookup/IdentityIdentity2Emb_month_vendor_created/embedding_lookup:output:0*
T0*D
_class:
86loc:@Emb_month_vendor_created/embedding_lookup/3550712*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
4Emb_month_vendor_created/embedding_lookup/Identity_1Identity;Emb_month_vendor_created/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   £
flatten/ReshapeReshape=Emb_month_vendor_created/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshape6Emb_Customer_City/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   £
flatten_2/ReshapeReshape9Emb_Customer_Country/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_3/ReshapeReshape3Emb_PrimaryTag/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_4/ReshapeReshape4Emb_Vendor_Tags/embedding_lookup/Identity_1:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-dense_layer_for_numbers/MatMul/ReadVariableOpReadVariableOp6dense_layer_for_numbers_matmul_readvariableop_resource*
_output_shapes

:Z<*
dtype0
dense_layer_for_numbers/MatMulMatMulinputs_55dense_layer_for_numbers/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¢
.dense_layer_for_numbers/BiasAdd/ReadVariableOpReadVariableOp7dense_layer_for_numbers_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0¾
dense_layer_for_numbers/BiasAddBiasAdd(dense_layer_for_numbers/MatMul:product:06dense_layer_for_numbers/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dense_layer_for_numbers/ReluRelu(dense_layer_for_numbers/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :±
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0*dense_layer_for_numbers/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
/dense_layer1_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer1_after_concat_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0²
 dense_layer1_after_concat/MatMulMatMulconcatenate/concat:output:07dense_layer1_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer1_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer1_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer1_after_concat/BiasAddBiasAdd*dense_layer1_after_concat/MatMul:product:08dense_layer1_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer1_after_concat/ReluRelu*dense_layer1_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
dropout1/IdentityIdentity,dense_layer1_after_concat/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/dense_layer2_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer2_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0±
 dense_layer2_after_concat/MatMulMatMuldropout1/Identity:output:07dense_layer2_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer2_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer2_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer2_after_concat/BiasAddBiasAdd*dense_layer2_after_concat/MatMul:product:08dense_layer2_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer2_after_concat/ReluRelu*dense_layer2_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
dropout2/IdentityIdentity,dense_layer2_after_concat/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/dense_layer3_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer3_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0±
 dense_layer3_after_concat/MatMulMatMuldropout2/Identity:output:07dense_layer3_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer3_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer3_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer3_after_concat/BiasAddBiasAdd*dense_layer3_after_concat/MatMul:product:08dense_layer3_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer3_after_concat/ReluRelu*dense_layer3_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output/MatMulMatMul,dense_layer3_after_concat/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp#^Emb_Customer_City/embedding_lookup&^Emb_Customer_Country/embedding_lookup ^Emb_PrimaryTag/embedding_lookup!^Emb_Vendor_Tags/embedding_lookup*^Emb_month_vendor_created/embedding_lookup^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp1^dense_layer1_after_concat/BiasAdd/ReadVariableOp0^dense_layer1_after_concat/MatMul/ReadVariableOp1^dense_layer2_after_concat/BiasAdd/ReadVariableOp0^dense_layer2_after_concat/MatMul/ReadVariableOp1^dense_layer3_after_concat/BiasAdd/ReadVariableOp0^dense_layer3_after_concat/MatMul/ReadVariableOp/^dense_layer_for_numbers/BiasAdd/ReadVariableOp.^dense_layer_for_numbers/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2H
"Emb_Customer_City/embedding_lookup"Emb_Customer_City/embedding_lookup2N
%Emb_Customer_Country/embedding_lookup%Emb_Customer_Country/embedding_lookup2B
Emb_PrimaryTag/embedding_lookupEmb_PrimaryTag/embedding_lookup2D
 Emb_Vendor_Tags/embedding_lookup Emb_Vendor_Tags/embedding_lookup2V
)Emb_month_vendor_created/embedding_lookup)Emb_month_vendor_created/embedding_lookup2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2d
0dense_layer1_after_concat/BiasAdd/ReadVariableOp0dense_layer1_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer1_after_concat/MatMul/ReadVariableOp/dense_layer1_after_concat/MatMul/ReadVariableOp2d
0dense_layer2_after_concat/BiasAdd/ReadVariableOp0dense_layer2_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer2_after_concat/MatMul/ReadVariableOp/dense_layer2_after_concat/MatMul/ReadVariableOp2d
0dense_layer3_after_concat/BiasAdd/ReadVariableOp0dense_layer3_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer3_after_concat/MatMul/ReadVariableOp/dense_layer3_after_concat/MatMul/ReadVariableOp2`
.dense_layer_for_numbers/BiasAdd/ReadVariableOp.dense_layer_for_numbers/BiasAdd/ReadVariableOp2^
-dense_layer_for_numbers/MatMul/ReadVariableOp-dense_layer_for_numbers/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
"
_user_specified_name
inputs/5
­


V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

0__inference_Emb_PrimaryTag_layer_call_fn_3550953

inputs
unknown:-
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
E
)__inference_flatten_layer_call_fn_3550985

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3549961`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
C__inference_Output_layer_call_and_return_conditional_losses_3550101

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
a
E__inference_dropout1_layer_call_and_return_conditional_losses_3550198

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®	
¬
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3550929

inputs*
embedding_lookup_3550923:7
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3550923Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3550923*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3550923*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶

3__inference_Emb_Customer_City_layer_call_fn_3550919

inputs
unknown:7
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­


V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3551096

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
p
°
B__inference_model_layer_call_and_return_conditional_losses_3550853
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5:
(emb_vendor_tags_embedding_lookup_3550776:G9
'emb_primarytag_embedding_lookup_3550782:-?
-emb_customer_country_embedding_lookup_3550788:<
*emb_customer_city_embedding_lookup_3550794:7C
1emb_month_vendor_created_embedding_lookup_3550800:H
6dense_layer_for_numbers_matmul_readvariableop_resource:Z<E
7dense_layer_for_numbers_biasadd_readvariableop_resource:<J
8dense_layer1_after_concat_matmul_readvariableop_resource:dG
9dense_layer1_after_concat_biasadd_readvariableop_resource:J
8dense_layer2_after_concat_matmul_readvariableop_resource:G
9dense_layer2_after_concat_biasadd_readvariableop_resource:J
8dense_layer3_after_concat_matmul_readvariableop_resource:G
9dense_layer3_after_concat_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity¢"Emb_Customer_City/embedding_lookup¢%Emb_Customer_Country/embedding_lookup¢Emb_PrimaryTag/embedding_lookup¢ Emb_Vendor_Tags/embedding_lookup¢)Emb_month_vendor_created/embedding_lookup¢Output/BiasAdd/ReadVariableOp¢Output/MatMul/ReadVariableOp¢0dense_layer1_after_concat/BiasAdd/ReadVariableOp¢/dense_layer1_after_concat/MatMul/ReadVariableOp¢0dense_layer2_after_concat/BiasAdd/ReadVariableOp¢/dense_layer2_after_concat/MatMul/ReadVariableOp¢0dense_layer3_after_concat/BiasAdd/ReadVariableOp¢/dense_layer3_after_concat/MatMul/ReadVariableOp¢.dense_layer_for_numbers/BiasAdd/ReadVariableOp¢-dense_layer_for_numbers/MatMul/ReadVariableOpg
Emb_Vendor_Tags/CastCastinputs_4*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý
 Emb_Vendor_Tags/embedding_lookupResourceGather(emb_vendor_tags_embedding_lookup_3550776Emb_Vendor_Tags/Cast:y:0*
Tindices0*;
_class1
/-loc:@Emb_Vendor_Tags/embedding_lookup/3550776*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0Ó
)Emb_Vendor_Tags/embedding_lookup/IdentityIdentity)Emb_Vendor_Tags/embedding_lookup:output:0*
T0*;
_class1
/-loc:@Emb_Vendor_Tags/embedding_lookup/3550776*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
+Emb_Vendor_Tags/embedding_lookup/Identity_1Identity2Emb_Vendor_Tags/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
Emb_PrimaryTag/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
Emb_PrimaryTag/embedding_lookupResourceGather'emb_primarytag_embedding_lookup_3550782Emb_PrimaryTag/Cast:y:0*
Tindices0*:
_class0
.,loc:@Emb_PrimaryTag/embedding_lookup/3550782*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ð
(Emb_PrimaryTag/embedding_lookup/IdentityIdentity(Emb_PrimaryTag/embedding_lookup:output:0*
T0*:
_class0
.,loc:@Emb_PrimaryTag/embedding_lookup/3550782*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*Emb_PrimaryTag/embedding_lookup/Identity_1Identity1Emb_PrimaryTag/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Emb_Customer_Country/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Emb_Customer_Country/embedding_lookupResourceGather-emb_customer_country_embedding_lookup_3550788Emb_Customer_Country/Cast:y:0*
Tindices0*@
_class6
42loc:@Emb_Customer_Country/embedding_lookup/3550788*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0â
.Emb_Customer_Country/embedding_lookup/IdentityIdentity.Emb_Customer_Country/embedding_lookup:output:0*
T0*@
_class6
42loc:@Emb_Customer_Country/embedding_lookup/3550788*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
0Emb_Customer_Country/embedding_lookup/Identity_1Identity7Emb_Customer_Country/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
Emb_Customer_City/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Emb_Customer_City/embedding_lookupResourceGather*emb_customer_city_embedding_lookup_3550794Emb_Customer_City/Cast:y:0*
Tindices0*=
_class3
1/loc:@Emb_Customer_City/embedding_lookup/3550794*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ù
+Emb_Customer_City/embedding_lookup/IdentityIdentity+Emb_Customer_City/embedding_lookup:output:0*
T0*=
_class3
1/loc:@Emb_Customer_City/embedding_lookup/3550794*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-Emb_Customer_City/embedding_lookup/Identity_1Identity4Emb_Customer_City/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Emb_month_vendor_created/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
)Emb_month_vendor_created/embedding_lookupResourceGather1emb_month_vendor_created_embedding_lookup_3550800!Emb_month_vendor_created/Cast:y:0*
Tindices0*D
_class:
86loc:@Emb_month_vendor_created/embedding_lookup/3550800*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0î
2Emb_month_vendor_created/embedding_lookup/IdentityIdentity2Emb_month_vendor_created/embedding_lookup:output:0*
T0*D
_class:
86loc:@Emb_month_vendor_created/embedding_lookup/3550800*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
4Emb_month_vendor_created/embedding_lookup/Identity_1Identity;Emb_month_vendor_created/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   £
flatten/ReshapeReshape=Emb_month_vendor_created/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshape6Emb_Customer_City/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   £
flatten_2/ReshapeReshape9Emb_Customer_Country/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_3/ReshapeReshape3Emb_PrimaryTag/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_4/ReshapeReshape4Emb_Vendor_Tags/embedding_lookup/Identity_1:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-dense_layer_for_numbers/MatMul/ReadVariableOpReadVariableOp6dense_layer_for_numbers_matmul_readvariableop_resource*
_output_shapes

:Z<*
dtype0
dense_layer_for_numbers/MatMulMatMulinputs_55dense_layer_for_numbers/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¢
.dense_layer_for_numbers/BiasAdd/ReadVariableOpReadVariableOp7dense_layer_for_numbers_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0¾
dense_layer_for_numbers/BiasAddBiasAdd(dense_layer_for_numbers/MatMul:product:06dense_layer_for_numbers/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dense_layer_for_numbers/ReluRelu(dense_layer_for_numbers/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :±
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0*dense_layer_for_numbers/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
/dense_layer1_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer1_after_concat_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0²
 dense_layer1_after_concat/MatMulMatMulconcatenate/concat:output:07dense_layer1_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer1_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer1_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer1_after_concat/BiasAddBiasAdd*dense_layer1_after_concat/MatMul:product:08dense_layer1_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer1_after_concat/ReluRelu*dense_layer1_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/dense_layer2_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer2_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 dense_layer2_after_concat/MatMulMatMul,dense_layer1_after_concat/Relu:activations:07dense_layer2_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer2_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer2_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer2_after_concat/BiasAddBiasAdd*dense_layer2_after_concat/MatMul:product:08dense_layer2_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer2_after_concat/ReluRelu*dense_layer2_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/dense_layer3_after_concat/MatMul/ReadVariableOpReadVariableOp8dense_layer3_after_concat_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 dense_layer3_after_concat/MatMulMatMul,dense_layer2_after_concat/Relu:activations:07dense_layer3_after_concat/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0dense_layer3_after_concat/BiasAdd/ReadVariableOpReadVariableOp9dense_layer3_after_concat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!dense_layer3_after_concat/BiasAddBiasAdd*dense_layer3_after_concat/MatMul:product:08dense_layer3_after_concat/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_layer3_after_concat/ReluRelu*dense_layer3_after_concat/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output/MatMulMatMul,dense_layer3_after_concat/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp#^Emb_Customer_City/embedding_lookup&^Emb_Customer_Country/embedding_lookup ^Emb_PrimaryTag/embedding_lookup!^Emb_Vendor_Tags/embedding_lookup*^Emb_month_vendor_created/embedding_lookup^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp1^dense_layer1_after_concat/BiasAdd/ReadVariableOp0^dense_layer1_after_concat/MatMul/ReadVariableOp1^dense_layer2_after_concat/BiasAdd/ReadVariableOp0^dense_layer2_after_concat/MatMul/ReadVariableOp1^dense_layer3_after_concat/BiasAdd/ReadVariableOp0^dense_layer3_after_concat/MatMul/ReadVariableOp/^dense_layer_for_numbers/BiasAdd/ReadVariableOp.^dense_layer_for_numbers/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2H
"Emb_Customer_City/embedding_lookup"Emb_Customer_City/embedding_lookup2N
%Emb_Customer_Country/embedding_lookup%Emb_Customer_Country/embedding_lookup2B
Emb_PrimaryTag/embedding_lookupEmb_PrimaryTag/embedding_lookup2D
 Emb_Vendor_Tags/embedding_lookup Emb_Vendor_Tags/embedding_lookup2V
)Emb_month_vendor_created/embedding_lookup)Emb_month_vendor_created/embedding_lookup2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2d
0dense_layer1_after_concat/BiasAdd/ReadVariableOp0dense_layer1_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer1_after_concat/MatMul/ReadVariableOp/dense_layer1_after_concat/MatMul/ReadVariableOp2d
0dense_layer2_after_concat/BiasAdd/ReadVariableOp0dense_layer2_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer2_after_concat/MatMul/ReadVariableOp/dense_layer2_after_concat/MatMul/ReadVariableOp2d
0dense_layer3_after_concat/BiasAdd/ReadVariableOp0dense_layer3_after_concat/BiasAdd/ReadVariableOp2b
/dense_layer3_after_concat/MatMul/ReadVariableOp/dense_layer3_after_concat/MatMul/ReadVariableOp2`
.dense_layer_for_numbers/BiasAdd/ReadVariableOp.dense_layer_for_numbers/BiasAdd/ReadVariableOp2^
-dense_layer_for_numbers/MatMul/ReadVariableOp-dense_layer_for_numbers/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
"
_user_specified_name
inputs/5

F
*__inference_dropout2_layer_call_fn_3551140

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550071`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
G
+__inference_flatten_4_layer_call_fn_3551029

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

¶+
#__inference__traced_restore_3551616
file_prefixF
4assignvariableop_emb_month_vendor_created_embeddings:A
/assignvariableop_1_emb_customer_city_embeddings:7D
2assignvariableop_2_emb_customer_country_embeddings:>
,assignvariableop_3_emb_primarytag_embeddings:-?
-assignvariableop_4_emb_vendor_tags_embeddings:GC
1assignvariableop_5_dense_layer_for_numbers_kernel:Z<=
/assignvariableop_6_dense_layer_for_numbers_bias:<E
3assignvariableop_7_dense_layer1_after_concat_kernel:d?
1assignvariableop_8_dense_layer1_after_concat_bias:E
3assignvariableop_9_dense_layer2_after_concat_kernel:@
2assignvariableop_10_dense_layer2_after_concat_bias:F
4assignvariableop_11_dense_layer3_after_concat_kernel:@
2assignvariableop_12_dense_layer3_after_concat_bias:3
!assignvariableop_13_output_kernel:-
assignvariableop_14_output_bias:'
assignvariableop_15_adam_iter:	 )
assignvariableop_16_adam_beta_1: )
assignvariableop_17_adam_beta_2: (
assignvariableop_18_adam_decay: 0
&assignvariableop_19_adam_learning_rate: #
assignvariableop_20_total: #
assignvariableop_21_count: 0
"assignvariableop_22_true_positives:1
#assignvariableop_23_false_negatives:2
$assignvariableop_24_true_positives_1:1
#assignvariableop_25_false_positives:.
$assignvariableop_26_true_positives_2: /
%assignvariableop_27_false_positives_1: /
%assignvariableop_28_false_negatives_1: 2
(assignvariableop_29_weights_intermediate: 2
$assignvariableop_30_true_positives_3:3
%assignvariableop_31_false_positives_2:3
%assignvariableop_32_false_negatives_2:8
*assignvariableop_33_weights_intermediate_1:P
>assignvariableop_34_adam_emb_month_vendor_created_embeddings_m:I
7assignvariableop_35_adam_emb_customer_city_embeddings_m:7L
:assignvariableop_36_adam_emb_customer_country_embeddings_m:F
4assignvariableop_37_adam_emb_primarytag_embeddings_m:-G
5assignvariableop_38_adam_emb_vendor_tags_embeddings_m:GK
9assignvariableop_39_adam_dense_layer_for_numbers_kernel_m:Z<E
7assignvariableop_40_adam_dense_layer_for_numbers_bias_m:<M
;assignvariableop_41_adam_dense_layer1_after_concat_kernel_m:dG
9assignvariableop_42_adam_dense_layer1_after_concat_bias_m:M
;assignvariableop_43_adam_dense_layer2_after_concat_kernel_m:G
9assignvariableop_44_adam_dense_layer2_after_concat_bias_m:M
;assignvariableop_45_adam_dense_layer3_after_concat_kernel_m:G
9assignvariableop_46_adam_dense_layer3_after_concat_bias_m::
(assignvariableop_47_adam_output_kernel_m:4
&assignvariableop_48_adam_output_bias_m:P
>assignvariableop_49_adam_emb_month_vendor_created_embeddings_v:I
7assignvariableop_50_adam_emb_customer_city_embeddings_v:7L
:assignvariableop_51_adam_emb_customer_country_embeddings_v:F
4assignvariableop_52_adam_emb_primarytag_embeddings_v:-G
5assignvariableop_53_adam_emb_vendor_tags_embeddings_v:GK
9assignvariableop_54_adam_dense_layer_for_numbers_kernel_v:Z<E
7assignvariableop_55_adam_dense_layer_for_numbers_bias_v:<M
;assignvariableop_56_adam_dense_layer1_after_concat_kernel_v:dG
9assignvariableop_57_adam_dense_layer1_after_concat_bias_v:M
;assignvariableop_58_adam_dense_layer2_after_concat_kernel_v:G
9assignvariableop_59_adam_dense_layer2_after_concat_bias_v:M
;assignvariableop_60_adam_dense_layer3_after_concat_kernel_v:G
9assignvariableop_61_adam_dense_layer3_after_concat_bias_v::
(assignvariableop_62_adam_output_kernel_v:4
&assignvariableop_63_adam_output_bias_v:
identity_65¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*È#
value¾#B»#AB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/3/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHõ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B æ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp4assignvariableop_emb_month_vendor_created_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp/assignvariableop_1_emb_customer_city_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_2AssignVariableOp2assignvariableop_2_emb_customer_country_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp,assignvariableop_3_emb_primarytag_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp-assignvariableop_4_emb_vendor_tags_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_5AssignVariableOp1assignvariableop_5_dense_layer_for_numbers_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_dense_layer_for_numbers_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_7AssignVariableOp3assignvariableop_7_dense_layer1_after_concat_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_8AssignVariableOp1assignvariableop_8_dense_layer1_after_concat_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_9AssignVariableOp3assignvariableop_9_dense_layer2_after_concat_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_10AssignVariableOp2assignvariableop_10_dense_layer2_after_concat_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_11AssignVariableOp4assignvariableop_11_dense_layer3_after_concat_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_12AssignVariableOp2assignvariableop_12_dense_layer3_after_concat_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_output_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_output_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_iterIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_negativesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp$assignvariableop_24_true_positives_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_positivesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp$assignvariableop_26_true_positives_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_false_positives_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_false_negatives_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_weights_intermediateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp$assignvariableop_30_true_positives_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_false_positives_2Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_false_negatives_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_weights_intermediate_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_emb_month_vendor_created_embeddings_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_emb_customer_city_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_36AssignVariableOp:assignvariableop_36_adam_emb_customer_country_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_emb_primarytag_embeddings_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_emb_vendor_tags_embeddings_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_39AssignVariableOp9assignvariableop_39_adam_dense_layer_for_numbers_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_dense_layer_for_numbers_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_41AssignVariableOp;assignvariableop_41_adam_dense_layer1_after_concat_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_42AssignVariableOp9assignvariableop_42_adam_dense_layer1_after_concat_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_43AssignVariableOp;assignvariableop_43_adam_dense_layer2_after_concat_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_44AssignVariableOp9assignvariableop_44_adam_dense_layer2_after_concat_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_45AssignVariableOp;assignvariableop_45_adam_dense_layer3_after_concat_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_46AssignVariableOp9assignvariableop_46_adam_dense_layer3_after_concat_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_output_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_output_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_emb_month_vendor_created_embeddings_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_emb_customer_city_embeddings_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_51AssignVariableOp:assignvariableop_51_adam_emb_customer_country_embeddings_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_emb_primarytag_embeddings_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_emb_vendor_tags_embeddings_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_54AssignVariableOp9assignvariableop_54_adam_dense_layer_for_numbers_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_dense_layer_for_numbers_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_adam_dense_layer1_after_concat_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_57AssignVariableOp9assignvariableop_57_adam_dense_layer1_after_concat_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_58AssignVariableOp;assignvariableop_58_adam_dense_layer2_after_concat_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_59AssignVariableOp9assignvariableop_59_adam_dense_layer2_after_concat_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_60AssignVariableOp;assignvariableop_60_adam_dense_layer3_after_concat_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_61AssignVariableOp9assignvariableop_61_adam_dense_layer3_after_concat_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_output_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp&assignvariableop_63_adam_output_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ï
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_65IdentityIdentity_64:output:0^NoOp_1*
T0*
_output_shapes
: ¼
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_65Identity_65:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­


V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3551135

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ø
c
E__inference_dropout1_layer_call_and_return_conditional_losses_3551111

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ	
³
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951

inputs*
embedding_lookup_3549945:
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_3549945Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3549945*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3549945*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¦
9__inference_dense_layer_for_numbers_layer_call_fn_3551044

inputs
unknown:Z<
	unknown_0:<
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿZ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs

F
*__inference_dropout1_layer_call_fn_3551101

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550047`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÎU
±

B__inference_model_layer_call_and_return_conditional_losses_3550108

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5)
emb_vendor_tags_3549896:G(
emb_primarytag_3549910:-.
emb_customer_country_3549924:+
emb_customer_city_3549938:72
 emb_month_vendor_created_3549952:1
dense_layer_for_numbers_3550007:Z<-
dense_layer_for_numbers_3550009:<3
!dense_layer1_after_concat_3550037:d/
!dense_layer1_after_concat_3550039:3
!dense_layer2_after_concat_3550061:/
!dense_layer2_after_concat_3550063:3
!dense_layer3_after_concat_3550085:/
!dense_layer3_after_concat_3550087: 
output_3550102:
output_3550104:
identity¢)Emb_Customer_City/StatefulPartitionedCall¢,Emb_Customer_Country/StatefulPartitionedCall¢&Emb_PrimaryTag/StatefulPartitionedCall¢'Emb_Vendor_Tags/StatefulPartitionedCall¢0Emb_month_vendor_created/StatefulPartitionedCall¢Output/StatefulPartitionedCall¢1dense_layer1_after_concat/StatefulPartitionedCall¢1dense_layer2_after_concat/StatefulPartitionedCall¢1dense_layer3_after_concat/StatefulPartitionedCall¢/dense_layer_for_numbers/StatefulPartitionedCallú
'Emb_Vendor_Tags/StatefulPartitionedCallStatefulPartitionedCallinputs_4emb_vendor_tags_3549896*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895÷
&Emb_PrimaryTag/StatefulPartitionedCallStatefulPartitionedCallinputs_3emb_primarytag_3549910*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909
,Emb_Customer_Country/StatefulPartitionedCallStatefulPartitionedCallinputs_2emb_customer_country_3549924*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923
)Emb_Customer_City/StatefulPartitionedCallStatefulPartitionedCallinputs_1emb_customer_city_3549938*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937
0Emb_month_vendor_created/StatefulPartitionedCallStatefulPartitionedCallinputs emb_month_vendor_created_3549952*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951ê
flatten/PartitionedCallPartitionedCall9Emb_month_vendor_created/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3549961ç
flatten_1/PartitionedCallPartitionedCall2Emb_Customer_City/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969ê
flatten_2/PartitionedCallPartitionedCall5Emb_Customer_Country/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977ä
flatten_3/PartitionedCallPartitionedCall/Emb_PrimaryTag/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985å
flatten_4/PartitionedCallPartitionedCall0Emb_Vendor_Tags/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993±
/dense_layer_for_numbers/StatefulPartitionedCallStatefulPartitionedCallinputs_5dense_layer_for_numbers_3550007dense_layer_for_numbers_3550009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006¨
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:08dense_layer_for_numbers/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023Õ
1dense_layer1_after_concat/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0!dense_layer1_after_concat_3550037!dense_layer1_after_concat_3550039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036í
dropout1/PartitionedCallPartitionedCall:dense_layer1_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550047Ò
1dense_layer2_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0!dense_layer2_after_concat_3550061!dense_layer2_after_concat_3550063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060í
dropout2/PartitionedCallPartitionedCall:dense_layer2_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550071Ò
1dense_layer3_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0!dense_layer3_after_concat_3550085!dense_layer3_after_concat_3550087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084
Output/StatefulPartitionedCallStatefulPartitionedCall:dense_layer3_after_concat/StatefulPartitionedCall:output:0output_3550102output_3550104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_3550101v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp*^Emb_Customer_City/StatefulPartitionedCall-^Emb_Customer_Country/StatefulPartitionedCall'^Emb_PrimaryTag/StatefulPartitionedCall(^Emb_Vendor_Tags/StatefulPartitionedCall1^Emb_month_vendor_created/StatefulPartitionedCall^Output/StatefulPartitionedCall2^dense_layer1_after_concat/StatefulPartitionedCall2^dense_layer2_after_concat/StatefulPartitionedCall2^dense_layer3_after_concat/StatefulPartitionedCall0^dense_layer_for_numbers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2V
)Emb_Customer_City/StatefulPartitionedCall)Emb_Customer_City/StatefulPartitionedCall2\
,Emb_Customer_Country/StatefulPartitionedCall,Emb_Customer_Country/StatefulPartitionedCall2P
&Emb_PrimaryTag/StatefulPartitionedCall&Emb_PrimaryTag/StatefulPartitionedCall2R
'Emb_Vendor_Tags/StatefulPartitionedCall'Emb_Vendor_Tags/StatefulPartitionedCall2d
0Emb_month_vendor_created/StatefulPartitionedCall0Emb_month_vendor_created/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2f
1dense_layer1_after_concat/StatefulPartitionedCall1dense_layer1_after_concat/StatefulPartitionedCall2f
1dense_layer2_after_concat/StatefulPartitionedCall1dense_layer2_after_concat/StatefulPartitionedCall2f
1dense_layer3_after_concat/StatefulPartitionedCall1dense_layer3_after_concat/StatefulPartitionedCall2b
/dense_layer_for_numbers/StatefulPartitionedCall/dense_layer_for_numbers/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
¬	
ª
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3550980

inputs*
embedding_lookup_3550974:G
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
embedding_lookupResourceGatherembedding_lookup_3550974Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/3550974*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/3550974*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
«


T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3551055

inputs0
matmul_readvariableop_resource:Z<-
biasadd_readvariableop_resource:<
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
æ
¨
;__inference_dense_layer1_after_concat_layer_call_fn_3551085

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

F
*__inference_dropout2_layer_call_fn_3551145

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550173`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«


T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006

inputs0
matmul_readvariableop_resource:Z<-
biasadd_readvariableop_resource:<
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
¬	
ª
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ÓV
Ú

B__inference_model_layer_call_and_return_conditional_losses_3550593
month_vendor_created
customer_city
customer_country
primary_tag
vendor_tags
numerical_inputs)
emb_vendor_tags_3550544:G(
emb_primarytag_3550547:-.
emb_customer_country_3550550:+
emb_customer_city_3550553:72
 emb_month_vendor_created_3550556:1
dense_layer_for_numbers_3550564:Z<-
dense_layer_for_numbers_3550566:<3
!dense_layer1_after_concat_3550570:d/
!dense_layer1_after_concat_3550572:3
!dense_layer2_after_concat_3550576:/
!dense_layer2_after_concat_3550578:3
!dense_layer3_after_concat_3550582:/
!dense_layer3_after_concat_3550584: 
output_3550587:
output_3550589:
identity¢)Emb_Customer_City/StatefulPartitionedCall¢,Emb_Customer_Country/StatefulPartitionedCall¢&Emb_PrimaryTag/StatefulPartitionedCall¢'Emb_Vendor_Tags/StatefulPartitionedCall¢0Emb_month_vendor_created/StatefulPartitionedCall¢Output/StatefulPartitionedCall¢1dense_layer1_after_concat/StatefulPartitionedCall¢1dense_layer2_after_concat/StatefulPartitionedCall¢1dense_layer3_after_concat/StatefulPartitionedCall¢/dense_layer_for_numbers/StatefulPartitionedCallý
'Emb_Vendor_Tags/StatefulPartitionedCallStatefulPartitionedCallvendor_tagsemb_vendor_tags_3550544*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895ú
&Emb_PrimaryTag/StatefulPartitionedCallStatefulPartitionedCallprimary_tagemb_primarytag_3550547*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3549909
,Emb_Customer_Country/StatefulPartitionedCallStatefulPartitionedCallcustomer_countryemb_customer_country_3550550*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3549923
)Emb_Customer_City/StatefulPartitionedCallStatefulPartitionedCallcustomer_cityemb_customer_city_3550553*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3549937¡
0Emb_month_vendor_created/StatefulPartitionedCallStatefulPartitionedCallmonth_vendor_created emb_month_vendor_created_3550556*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3549951ê
flatten/PartitionedCallPartitionedCall9Emb_month_vendor_created/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3549961ç
flatten_1/PartitionedCallPartitionedCall2Emb_Customer_City/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969ê
flatten_2/PartitionedCallPartitionedCall5Emb_Customer_Country/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977ä
flatten_3/PartitionedCallPartitionedCall/Emb_PrimaryTag/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_3549985å
flatten_4/PartitionedCallPartitionedCall0Emb_Vendor_Tags/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_3549993¹
/dense_layer_for_numbers/StatefulPartitionedCallStatefulPartitionedCallnumerical_inputsdense_layer_for_numbers_3550564dense_layer_for_numbers_3550566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3550006¨
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:08dense_layer_for_numbers/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023Õ
1dense_layer1_after_concat/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0!dense_layer1_after_concat_3550570!dense_layer1_after_concat_3550572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3550036í
dropout1/PartitionedCallPartitionedCall:dense_layer1_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout1_layer_call_and_return_conditional_losses_3550198Ò
1dense_layer2_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0!dense_layer2_after_concat_3550576!dense_layer2_after_concat_3550578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3550060í
dropout2/PartitionedCallPartitionedCall:dense_layer2_after_concat/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout2_layer_call_and_return_conditional_losses_3550173Ò
1dense_layer3_after_concat/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0!dense_layer3_after_concat_3550582!dense_layer3_after_concat_3550584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084
Output/StatefulPartitionedCallStatefulPartitionedCall:dense_layer3_after_concat/StatefulPartitionedCall:output:0output_3550587output_3550589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_3550101v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp*^Emb_Customer_City/StatefulPartitionedCall-^Emb_Customer_Country/StatefulPartitionedCall'^Emb_PrimaryTag/StatefulPartitionedCall(^Emb_Vendor_Tags/StatefulPartitionedCall1^Emb_month_vendor_created/StatefulPartitionedCall^Output/StatefulPartitionedCall2^dense_layer1_after_concat/StatefulPartitionedCall2^dense_layer2_after_concat/StatefulPartitionedCall2^dense_layer3_after_concat/StatefulPartitionedCall0^dense_layer_for_numbers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¥
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿZ: : : : : : : : : : : : : : : 2V
)Emb_Customer_City/StatefulPartitionedCall)Emb_Customer_City/StatefulPartitionedCall2\
,Emb_Customer_Country/StatefulPartitionedCall,Emb_Customer_Country/StatefulPartitionedCall2P
&Emb_PrimaryTag/StatefulPartitionedCall&Emb_PrimaryTag/StatefulPartitionedCall2R
'Emb_Vendor_Tags/StatefulPartitionedCall'Emb_Vendor_Tags/StatefulPartitionedCall2d
0Emb_month_vendor_created/StatefulPartitionedCall0Emb_month_vendor_created/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2f
1dense_layer1_after_concat/StatefulPartitionedCall1dense_layer1_after_concat/StatefulPartitionedCall2f
1dense_layer2_after_concat/StatefulPartitionedCall1dense_layer2_after_concat/StatefulPartitionedCall2f
1dense_layer3_after_concat/StatefulPartitionedCall1dense_layer3_after_concat/StatefulPartitionedCall2b
/dense_layer_for_numbers/StatefulPartitionedCall/dense_layer_for_numbers/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namemonth_vendor_created:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecustomer_city:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namecustomer_country:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameprimary_tag:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%
_user_specified_namevendor_tags:YU
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
*
_user_specified_namenumerical_inputs
©
G
+__inference_flatten_2_layer_call_fn_3551007

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_3549977`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_3550991

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨


-__inference_concatenate_layer_call_fn_3551065
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityì
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3550023`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ<:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
"
_user_specified_name
inputs/5
æ
¨
;__inference_dense_layer3_after_concat_layer_call_fn_3551163

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3550084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

1__inference_Emb_Vendor_Tags_layer_call_fn_3550970

inputs
unknown:G
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3549895s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
©
G
+__inference_flatten_1_layer_call_fn_3550996

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_3549969`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
a
E__inference_dropout2_layer_call_and_return_conditional_losses_3550173

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_3551002

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
G
customer_city6
serving_default_customer_city:0ÿÿÿÿÿÿÿÿÿ
M
customer_country9
"serving_default_customer_country:0ÿÿÿÿÿÿÿÿÿ
U
month_vendor_created=
&serving_default_month_vendor_created:0ÿÿÿÿÿÿÿÿÿ
M
numerical_inputs9
"serving_default_numerical_inputs:0ÿÿÿÿÿÿÿÿÿZ
C
primary_tag4
serving_default_primary_tag:0ÿÿÿÿÿÿÿÿÿ
C
vendor_tags4
serving_default_vendor_tags:0ÿÿÿÿÿÿÿÿÿ
:
Output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÜÃ
­
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer_with_weights-7
layer-20
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
µ
"
embeddings
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
)
embeddings
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
0
embeddings
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
7
embeddings
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
>
embeddings
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
¥
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
»

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}_random_generator
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

	iter
 beta_1
¡beta_2

¢decay
£learning_rate"m¥)m¦0m§7m¨>m©cmªdm«qm¬rm­	m®	m¯	m°	m±	m²	m³"v´)vµ0v¶7v·>v¸cv¹dvºqv»rv¼	v½	v¾	v¿	vÀ	vÁ	vÂ"
	optimizer

"0
)1
02
73
>4
c5
d6
q7
r8
9
10
11
12
13
14"
trackable_list_wrapper

"0
)1
02
73
>4
c5
d6
q7
r8
9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ê2ç
'__inference_model_layer_call_fn_3550141
'__inference_model_layer_call_fn_3550639
'__inference_model_layer_call_fn_3550679
'__inference_model_layer_call_fn_3550479À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_3550767
B__inference_model_layer_call_and_return_conditional_losses_3550853
B__inference_model_layer_call_and_return_conditional_losses_3550536
B__inference_model_layer_call_and_return_conditional_losses_3550593À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
§B¤
"__inference__wrapped_model_3549868month_vendor_createdcustomer_citycustomer_countryprimary_tagvendor_tagsnumerical_inputs"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
©serving_default"
signature_map
5:32#Emb_month_vendor_created/embeddings
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ä2á
:__inference_Emb_month_vendor_created_layer_call_fn_3550902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿ2ü
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3550912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.:,72Emb_Customer_City/embeddings
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_Emb_Customer_City_layer_call_fn_3550919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3550929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
1:/2Emb_Customer_Country/embeddings
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
à2Ý
6__inference_Emb_Customer_Country_layer_call_fn_3550936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3550946¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)-2Emb_PrimaryTag/embeddings
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_Emb_PrimaryTag_layer_call_fn_3550953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3550963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*G2Emb_Vendor_Tags/embeddings
'
>0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_Emb_Vendor_Tags_layer_call_fn_3550970¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3550980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_layer_call_fn_3550985¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_3550991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_1_layer_call_fn_3550996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_1_layer_call_and_return_conditional_losses_3551002¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_2_layer_call_fn_3551007¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_2_layer_call_and_return_conditional_losses_3551013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_3_layer_call_fn_3551018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_3_layer_call_and_return_conditional_losses_3551024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_4_layer_call_fn_3551029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_4_layer_call_and_return_conditional_losses_3551035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0:.Z<2dense_layer_for_numbers/kernel
*:(<2dense_layer_for_numbers/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ã2à
9__inference_dense_layer_for_numbers_layer_call_fn_3551044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3551055¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_concatenate_layer_call_fn_3551065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_concatenate_layer_call_and_return_conditional_losses_3551076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2:0d2 dense_layer1_after_concat/kernel
,:*2dense_layer1_after_concat/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
å2â
;__inference_dense_layer1_after_concat_layer_call_fn_3551085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3551096¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout1_layer_call_fn_3551101
*__inference_dropout1_layer_call_fn_3551106´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout1_layer_call_and_return_conditional_losses_3551111
E__inference_dropout1_layer_call_and_return_conditional_losses_3551115´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2:02 dense_layer2_after_concat/kernel
,:*2dense_layer2_after_concat/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
å2â
;__inference_dense_layer2_after_concat_layer_call_fn_3551124¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3551135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout2_layer_call_fn_3551140
*__inference_dropout2_layer_call_fn_3551145´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout2_layer_call_and_return_conditional_losses_3551150
E__inference_dropout2_layer_call_and_return_conditional_losses_3551154´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2:02 dense_layer3_after_concat/kernel
,:*2dense_layer3_after_concat/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
å2â
;__inference_dense_layer3_after_concat_layer_call_fn_3551163¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3551174¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:2Output/kernel
:2Output/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_Output_layer_call_fn_3551183¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_Output_layer_call_and_return_conditional_losses_3551194¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
Ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
H
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B¡
%__inference_signature_wrapper_3550895customer_citycustomer_countrymonth_vendor_creatednumerical_inputsprimary_tagvendor_tags"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
v

thresholds
true_positives
false_negatives
	variables
	keras_api"
_tf_keras_metric
v

thresholds
true_positives
false_positives
	variables
	keras_api"
_tf_keras_metric
§

init_shape
true_positives
false_positives
false_negatives
weights_intermediate
	variables
	keras_api"
_tf_keras_metric
§

init_shape
true_positives
 false_positives
¡false_negatives
¢weights_intermediate
£	variables
¤	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2weights_intermediate
@
0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2weights_intermediate
@
0
 1
¡2
¢3"
trackable_list_wrapper
.
£	variables"
_generic_user_object
::82*Adam/Emb_month_vendor_created/embeddings/m
3:172#Adam/Emb_Customer_City/embeddings/m
6:42&Adam/Emb_Customer_Country/embeddings/m
0:.-2 Adam/Emb_PrimaryTag/embeddings/m
1:/G2!Adam/Emb_Vendor_Tags/embeddings/m
5:3Z<2%Adam/dense_layer_for_numbers/kernel/m
/:-<2#Adam/dense_layer_for_numbers/bias/m
7:5d2'Adam/dense_layer1_after_concat/kernel/m
1:/2%Adam/dense_layer1_after_concat/bias/m
7:52'Adam/dense_layer2_after_concat/kernel/m
1:/2%Adam/dense_layer2_after_concat/bias/m
7:52'Adam/dense_layer3_after_concat/kernel/m
1:/2%Adam/dense_layer3_after_concat/bias/m
$:"2Adam/Output/kernel/m
:2Adam/Output/bias/m
::82*Adam/Emb_month_vendor_created/embeddings/v
3:172#Adam/Emb_Customer_City/embeddings/v
6:42&Adam/Emb_Customer_Country/embeddings/v
0:.-2 Adam/Emb_PrimaryTag/embeddings/v
1:/G2!Adam/Emb_Vendor_Tags/embeddings/v
5:3Z<2%Adam/dense_layer_for_numbers/kernel/v
/:-<2#Adam/dense_layer_for_numbers/bias/v
7:5d2'Adam/dense_layer1_after_concat/kernel/v
1:/2%Adam/dense_layer1_after_concat/bias/v
7:52'Adam/dense_layer2_after_concat/kernel/v
1:/2%Adam/dense_layer2_after_concat/bias/v
7:52'Adam/dense_layer3_after_concat/kernel/v
1:/2%Adam/dense_layer3_after_concat/bias/v
$:"2Adam/Output/kernel/v
:2Adam/Output/bias/v±
N__inference_Emb_Customer_City_layer_call_and_return_conditional_losses_3550929_)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_Emb_Customer_City_layer_call_fn_3550919R)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ´
Q__inference_Emb_Customer_Country_layer_call_and_return_conditional_losses_3550946_0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_Emb_Customer_Country_layer_call_fn_3550936R0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
K__inference_Emb_PrimaryTag_layer_call_and_return_conditional_losses_3550963_7/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Emb_PrimaryTag_layer_call_fn_3550953R7/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
L__inference_Emb_Vendor_Tags_layer_call_and_return_conditional_losses_3550980_>/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
1__inference_Emb_Vendor_Tags_layer_call_fn_3550970R>/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
¸
U__inference_Emb_month_vendor_created_layer_call_and_return_conditional_losses_3550912_"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
:__inference_Emb_month_vendor_created_layer_call_fn_3550902R"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_Output_layer_call_and_return_conditional_losses_3551194^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_Output_layer_call_fn_3551183Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
"__inference__wrapped_model_3549868á>70)"cdqr¢
¢
ÿ
.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
'$
customer_cityÿÿÿÿÿÿÿÿÿ
*'
customer_countryÿÿÿÿÿÿÿÿÿ
%"
primary_tagÿÿÿÿÿÿÿÿÿ
%"
vendor_tagsÿÿÿÿÿÿÿÿÿ

*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
ª "/ª,
*
Output 
Outputÿÿÿÿÿÿÿÿÿæ
H__inference_concatenate_layer_call_and_return_conditional_losses_3551076ï¢ë
ã¢ß
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ<
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¾
-__inference_concatenate_layer_call_fn_3551065ï¢ë
ã¢ß
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿd¶
V__inference_dense_layer1_after_concat_layer_call_and_return_conditional_losses_3551096\qr/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
;__inference_dense_layer1_after_concat_layer_call_fn_3551085Oqr/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¸
V__inference_dense_layer2_after_concat_layer_call_and_return_conditional_losses_3551135^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
;__inference_dense_layer2_after_concat_layer_call_fn_3551124Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¸
V__inference_dense_layer3_after_concat_layer_call_and_return_conditional_losses_3551174^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
;__inference_dense_layer3_after_concat_layer_call_fn_3551163Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ´
T__inference_dense_layer_for_numbers_layer_call_and_return_conditional_losses_3551055\cd/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿZ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ<
 
9__inference_dense_layer_for_numbers_layer_call_fn_3551044Ocd/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿZ
ª "ÿÿÿÿÿÿÿÿÿ<¥
E__inference_dropout1_layer_call_and_return_conditional_losses_3551111\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout1_layer_call_and_return_conditional_losses_3551115\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dropout1_layer_call_fn_3551101O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout1_layer_call_fn_3551106O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dropout2_layer_call_and_return_conditional_losses_3551150\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout2_layer_call_and_return_conditional_losses_3551154\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dropout2_layer_call_fn_3551140O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout2_layer_call_fn_3551145O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_flatten_1_layer_call_and_return_conditional_losses_3551002\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_flatten_1_layer_call_fn_3550996O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_flatten_2_layer_call_and_return_conditional_losses_3551013\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_flatten_2_layer_call_fn_3551007O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_flatten_3_layer_call_and_return_conditional_losses_3551024\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_flatten_3_layer_call_fn_3551018O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_flatten_4_layer_call_and_return_conditional_losses_3551035\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_flatten_4_layer_call_fn_3551029O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_flatten_layer_call_and_return_conditional_losses_3550991\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_flatten_layer_call_fn_3550985O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
B__inference_model_layer_call_and_return_conditional_losses_3550536ß>70)"cdqr¢
¢
ÿ
.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
'$
customer_cityÿÿÿÿÿÿÿÿÿ
*'
customer_countryÿÿÿÿÿÿÿÿÿ
%"
primary_tagÿÿÿÿÿÿÿÿÿ
%"
vendor_tagsÿÿÿÿÿÿÿÿÿ

*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
B__inference_model_layer_call_and_return_conditional_losses_3550593ß>70)"cdqr¢
¢
ÿ
.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
'$
customer_cityÿÿÿÿÿÿÿÿÿ
*'
customer_countryÿÿÿÿÿÿÿÿÿ
%"
primary_tagÿÿÿÿÿÿÿÿÿ
%"
vendor_tagsÿÿÿÿÿÿÿÿÿ

*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ÿ
B__inference_model_layer_call_and_return_conditional_losses_3550767¸>70)"cdqr÷¢ó
ë¢ç
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ

"
inputs/5ÿÿÿÿÿÿÿÿÿZ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ÿ
B__inference_model_layer_call_and_return_conditional_losses_3550853¸>70)"cdqr÷¢ó
ë¢ç
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ

"
inputs/5ÿÿÿÿÿÿÿÿÿZ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 þ
'__inference_model_layer_call_fn_3550141Ò>70)"cdqr¢
¢
ÿ
.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
'$
customer_cityÿÿÿÿÿÿÿÿÿ
*'
customer_countryÿÿÿÿÿÿÿÿÿ
%"
primary_tagÿÿÿÿÿÿÿÿÿ
%"
vendor_tagsÿÿÿÿÿÿÿÿÿ

*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
p 

 
ª "ÿÿÿÿÿÿÿÿÿþ
'__inference_model_layer_call_fn_3550479Ò>70)"cdqr¢
¢
ÿ
.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
'$
customer_cityÿÿÿÿÿÿÿÿÿ
*'
customer_countryÿÿÿÿÿÿÿÿÿ
%"
primary_tagÿÿÿÿÿÿÿÿÿ
%"
vendor_tagsÿÿÿÿÿÿÿÿÿ

*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
p

 
ª "ÿÿÿÿÿÿÿÿÿ×
'__inference_model_layer_call_fn_3550639«>70)"cdqr÷¢ó
ë¢ç
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ

"
inputs/5ÿÿÿÿÿÿÿÿÿZ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ×
'__inference_model_layer_call_fn_3550679«>70)"cdqr÷¢ó
ë¢ç
ÜØ
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ

"
inputs/5ÿÿÿÿÿÿÿÿÿZ
p

 
ª "ÿÿÿÿÿÿÿÿÿó
%__inference_signature_wrapper_3550895É>70)"cdqrþ¢ú
¢ 
òªî
8
customer_city'$
customer_cityÿÿÿÿÿÿÿÿÿ
>
customer_country*'
customer_countryÿÿÿÿÿÿÿÿÿ
F
month_vendor_created.+
month_vendor_createdÿÿÿÿÿÿÿÿÿ
>
numerical_inputs*'
numerical_inputsÿÿÿÿÿÿÿÿÿZ
4
primary_tag%"
primary_tagÿÿÿÿÿÿÿÿÿ
4
vendor_tags%"
vendor_tagsÿÿÿÿÿÿÿÿÿ
"/ª,
*
Output 
Outputÿÿÿÿÿÿÿÿÿ