# MARKER

范式中的 marker 设计说明

- 所有范式的Marker都在名为 `ParadigmMarker` 的流中
- 不同范式以 `<prefix>_EXPERIMENT_START ` 开始, 以 `<prefix>_EXPERIMENT_END ` 结束
  - `<prefix>` 为范式的前缀, 同一个范式的 marker 都含有该前缀
  - [PRT](#PRT) 范式有一些不同

## emotion face

以 `EMF` 为前缀

每个 Block 开始时会发送 `<prefix>_BLOCK_{block_idx}_START ` , 结束时发送 `<prefix>_BLOCK_END`

- {block_idx} 为当前 block 的 id, 一般从 0 开始

每个 Trial 会发送 `<prefix>_TRIAL_START_{emotion_label}_{intensity_label} `

- emotion_label 为当前试次刺激的情绪标签, 分为 `positive`, `negative`, `neutral`
- intensity_label 为当前试次刺激的强度标签, 取值范围为[0,9]

根据受试的响应情况会发送 `<prefix>_RESPONSE_{resp} `或者 `<prefix>_NORESPONSE `

- resp 代表受试判断刺激所属的情绪
- NORESPONSE 代表受试反应超时

随后可能会有 `<prefix>_RESPONSE_RATING_{intensity} `或者 `<prefix>_RESPONSE_RATING_NEUTRAL`

- 根据受试的 resp 情况会有以上两种可能, 第二种(NEUTRAL)代表受试判断为中性, 所以不需要进行强度判断
- intensity 代表受试对该刺激的情绪强度进行的判读
  - 精度为小数点后两位

### 其他

emotion face 中会夹杂眼动校准, 也会发送一些 marker, 不必处理

## PRT

以 `PRT `为前缀

该范式开始时以 `<prefix>_EXPERIMENT_START_{high_side} ` 开始

- high_side 为该范式过程中的高奖励刺激的类型, 可选值为 `long` 和 `short`

每个 Block 开始时会发送 `<prefix>_BLOCK_{block_idx}_START ` , 结束时发送 `<prefix>_BLOCK_END`

- {block_idx} 为当前 block 的 id, 一般从 0 开始

每个 Trial 会发送 `<prefix>_TRIAL_START_{side} `

- side 为当前试次刺激的类型, 可选值为 `long` 和 `short`

根据受试的响应情况会发送 `<prefix>_RESPONSE_{resp} `或者 `<prefix>_NORESPONSE `

- resp 代表受试的按键, 可选值为 `s` 和 `l `分别代表 `short`和 `long`
- NORESPONSE 代表受试反应超时

## RESTING

以 `RESTING`为前缀

闭眼和睁眼阶段开始和结束时分别会发送 `<prefix>_{state}_START `和 `<prefix>_{state}_END`

- state 可选值为 `EYE_CLOSE` 和 `EYE_OPEN`

## SRET

以 `SRET`为前缀

Block 开始时会发送 `<prefix>_ENCODING_PHASE_START ` , 结束时发送 `<prefix>_ENCODING_PHASE_END`

每个 Trial 会发送 `<prefix>_ENCODING_STIM_ONSET_{trial} `

- trial 为该试次显示的词语(刺激)

根据受试的响应情况会发送 `<prefix>_RESPONSE_{resp} `或者 `<prefix>_NORESPONSE `

- resp 代表受试判断的该词语 `是否`符合自己, 可选值为 `yes`和 `no`
- NORESPONSE 代表受试反应超时

如果受试有进行响应, 随后会有 `<prefix>_RESPONSE_RATING_{intensity}`

- intensity 代表受试对该刺激的符合/不符合自己的程度的评分
- 精度为小数点后两位
