# 给朋友的迁移指南（OBJ `base + delta`）

这份文档是给“接手网页的同学”用的，按步骤做就能把旧版“每个卡片加载一个大 OBJ”，迁移到新版“`base.obj + delta_*.obj`”。

## 1. 迁移目标

- 旧方式：一个结果对应一个完整 OBJ（体积大、加载慢）
- 新方式：一个场景共享一个 `base.obj`，每个结果只加载自己的 `delta_*.obj`
- 效果：画面保持一致，但总体体积和加载压力显著下降

---

## 2. 你需要拿到哪些文件

每个场景目录都应包含以下文件（示例）：

```text
0a7cc_optimized/
  manifest.json
  base.obj
  delta_GT.obj
  delta_spatialllm.obj
  delta_ss.obj
  delta_llava3d.obj
  delta_ours.obj
  split_report.json
```

另一个场景 `09d6e808b4_optimized/` 同理。

---

## 3. 前端要改什么（最少改动）

### 3.1 数据配置改成 manifest 驱动

不要再手写完整 OBJ 名称列表，改为：

- 每一行只配置 `folder` 和 `manifest`
- 运行时读取 `manifest.json`，得到 `base` 和 `variants[*].delta`

### 3.2 单模型加载改成组合加载

每个卡片加载两份 OBJ：

- `baseUrl = ./${folder}/base.obj`
- `deltaUrl = ./${folder}/delta_xxx.obj`

然后放进同一个 `THREE.Group` 渲染。

### 3.3 给 base 增加缓存

同一场景多个卡片共享同一个 `base.obj`，建议缓存 Promise：

```js
const basePromiseCache = new Map();
```

避免重复下载和重复解析。

---

## 4. 已实现参考（本仓库）

如果你朋友直接参考本项目，关键文件如下：

- `viewer.js`：已完成 `manifest + base/delta + base缓存`
- `index.html`：已更新 section 提示文案
- `WEB_BASE_DELTA_MIGRATION.md`：更详细的技术说明

---

## 5. 如何重新生成优化数据（数据维护）

在项目根目录执行：

```bash
python build_obj_base_delta.py
```

会自动输出到：

- `0a7cc_optimized/`
- `09d6e808b4_optimized/`

---

## 6. 如何启动网页

在项目根目录启动静态服务器：

```bash
python -m http.server 8000
```

浏览器打开：

- `http://localhost:8000/index.html`

不要直接双击 HTML 打开（会触发本地跨域限制，OBJ 加载失败）。

---

## 7. 验收清单

- 10 个卡片都能显示
- 视觉效果与旧版一致
- 控制台没有 404 和 OBJ 解析报错
- 同场景 `base.obj` 不重复请求（缓存生效）

---

## 8. 常见问题

- 看起来“错位”  
  - 检查 `base` 和 `delta` 是否来自同一次拆分产物，不要混用不同批次文件。

- 颜色异常  
  - 当前方案依赖 OBJ 顶点色；如果你的数据没有顶点色，需要在材质里设置默认颜色。

- 页面空白  
  - 基本是路径错或没用静态服务器，先看 Network 和 Console。
