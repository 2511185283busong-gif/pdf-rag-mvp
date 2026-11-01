#!/usr/bin/env python3
"""Write the fictional Chinese manual used by zh_manual_cases.json as HTML, one section per page."""
import sys
S=[
("第一章 系统概述",["星河调度系统（StarRoute）负责把电商订单分配给同城配送司机。系统由四个服务组成：订单接入服务 gateway、路径规划服务 planner、司机匹配服务 matcher 和回执服务 receipt。","系统部署在两个数据中心：杭州为主站点，承载全部线上流量；成都为备用站点，平时只同步数据，不接收用户请求。","订单进入 gateway 后先写入 Kafka，再由 planner 计算路线，最后由 matcher 选择司机。司机完成配送后，receipt 负责把回执写回订单系统。"]),
("第二章 部署要求",["生产环境统一使用 Ubuntu 22.04。各服务的最低配置不同：planner 需要 16 核 CPU 和 64GB 内存，matcher 需要 8 核 CPU 和 32GB 内存，gateway 需要 4 核 CPU 和 8GB 内存。","每个节点至少配备 500GB SSD。不允许使用机械硬盘，否则路线缓存的写入延迟会明显升高。","新节点上线前必须运行 starctl precheck 完成环境检查，检查不通过的节点不得加入集群。"]),
("第三章 配置文件",["主配置文件位于 /etc/starroute/app.yaml。其中 max_batch_size 控制 planner 单批处理的订单数量，默认值为 200；log_level 控制日志级别，默认值为 info。","修改配置后执行 starctl reload 即可生效，不需要重启服务。只有修改监听端口 listen_port 时才必须重启。","配置文件由 Git 仓库统一管理，禁止直接在服务器上手工修改后不提交。"]),
("第四章 超时与重试",["gateway 调用 planner 的超时时间为 3 秒，失败后重试 2 次。planner 调用地图 API 的超时时间为 800 毫秒，失败后重试 1 次。matcher 查询司机定位的超时时间为 5 秒，失败后不重试，直接使用上一次的定位。","所有重试都采用指数退避，起始等待时间为 200 毫秒。","如果调整这些数值，需要同时更新压测基线，否则容量评估会失真。"]),
("第五章 告警等级",["告警分为三个等级，等级越小越紧急。下表列出每个等级的响应时间（分钟）和升级时通知的人数。","TABLE","P1 告警表示线上订单无法处理，P2 告警表示部分功能降级，P3 告警表示存在风险但暂未影响用户。超过响应时间未确认的告警会自动升级。"]),
("第六章 值班安排",["值班每周轮换一次，交接时间为每周一上午 10 点。每个班次安排一名主值班和一名副值班，节假日期间必须两人同时在线。","主值班负责确认告警并组织处理，副值班负责记录时间线和对外通报。值班沟通统一使用“星河-值班”群。","交接时必须说明未关闭的告警和正在进行的变更。"]),
("第七章 数据备份",["MySQL 每天凌晨 2 点 30 分执行一次全量备份，每 15 分钟归档一次 binlog。备份文件上传到对象存储，保留 30 天。","每个季度必须做一次恢复演练，演练结果记录在运维知识库。没有通过恢复演练的备份方案不得视为有效。","Kafka 的数据不做长期备份，消息保留时间为 7 天。"]),
("第八章 故障切换",["当杭州站点不可用超过 10 分钟，并且值班经理批准后，可以切换到成都站点。切换命令为 starctl failover --to chengdu。","切换的恢复时间目标（RTO）为 20 分钟，恢复点目标（RPO）为 1 分钟。","故障恢复后切回杭州必须安排在夜间低峰期，也就是凌晨 1 点到 5 点之间。"]),
("第九章 容量规划",["单个 planner 节点每秒最多处理 120 个订单。当 CPU 使用率持续超过 70% 达到 30 分钟时，需要扩容。","每年双十一之前 14 天必须完成一次全链路压测，压测目标是日常峰值的 3 倍。","扩容后要观察至少一个完整的业务高峰，确认负载已经均衡。"]),
("第十章 安全规范",["所有密钥都存放在 Vault 中，每 90 天轮换一次，禁止写入代码仓库。","运维人员登录生产服务器必须经过堡垒机，并完成二次验证。访问日志保留 180 天，供安全审计使用。","离职人员的权限必须在当天回收。"]),
("第十一章 常见故障处理",["订单堆积：先查看 Kafka 消费延迟，如果积压超过 5000 条，就扩容 matcher。","地图 API 限流：当出现 HTTP 429 时，执行 starctl cache on 启用本地路径缓存，这个模式最多持续 2 小时，之后必须关闭，否则路线会过期。","司机定位缺失：检查定位上报服务是否正常，并通知司机端负责人。"]),
("第十二章 变更流程",["生产变更只能在工作日下午 2 点到 5 点之间进行，周五不允许发布。","发布采用灰度方式：先放量 5%，观察 30 分钟，再放量到 50%，最后全量。发现异常时执行 starctl rollback 回滚，回滚应在 5 分钟内完成。","所有变更都必须有审批单，并由另一名工程师复核。"]),
]
SV=["计费","通知","库存","地址","优惠券","风控","客服","结算","报表","仓储","培训","电子围栏","天气","评价","保险","发票","合同","排班","车辆","充电","短信","语音","画像","审计"]
for i,n in enumerate(SV):
    S.append((f"附录 {i+1} {n}服务运行手册",[f"{n}服务由{n}团队负责，默认监听端口为 {8100+i}，线上运行 {2+i%5} 个副本。",f"{n}服务调用下游接口的超时时间为 {i%7+1} 秒，错误率超过 {i%4+1}% 时触发告警。",f"{n}服务的运行日志保留 {7+i*3} 天，发布前需要{n}团队负责人确认。"]))
T='<table><tr><th>等级</th><th>响应时间</th><th>通知人数</th></tr><tr><td>1</td><td>5</td><td>3</td></tr><tr><td>2</td><td>15</td><td>2</td></tr><tr><td>3</td><td>60</td><td>1</td></tr></table>'
h=['<!doctype html><meta charset=utf-8><style>@page{size:A4;margin:0}body{margin:0;font:15px/1.9 "PingFang SC","Songti SC",sans-serif}.p{height:296mm;box-sizing:border-box;padding:18mm 22mm;position:relative;break-after:page;overflow:hidden}.h{color:#666;font-size:12px;position:absolute;top:8mm;left:22mm}.f{color:#666;font-size:12px;position:absolute;bottom:8mm;left:22mm}h1{font-size:22px;margin:8mm 0 6mm}p{margin:0 0 7mm}table{border-collapse:collapse;margin:0 0 7mm}td,th{border:1px solid #888;padding:3px 26px}</style>']
for i,(t,ps) in enumerate(S,1):
    h.append(f'<div class=p><div class=h>星河调度系统运维手册</div><h1>{t}</h1>'+''.join(T if x=="TABLE" else f'<p>{x}</p>' for x in ps)+f'<div class=f>第 {i} 页</div></div>')
open(sys.argv[1],"w",encoding="utf-8").write("".join(h))
