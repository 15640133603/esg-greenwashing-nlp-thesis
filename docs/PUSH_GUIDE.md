# GitHub 发布指南

## 方式一：自动推送（推荐，需先授权）

在 WorkBuddy/CodeBuddy 设置页 →「连接器」中授权 GitHub 后，告诉我"已授权"，
我会自动完成建仓与推送，无需任何手动操作。

## 方式二：手动推送（备用）

1. 下载仓库压缩包 `esg-greenwashing-nlp.zip`（已放在工作区根目录）
2. 解压后在仓库目录执行：

```bash
cd esg-greenwashing-nlp

# 初始化（若已含 .git 目录可跳过 init/add/commit）
git init -b main
git add -A
git commit -m "初始提交：企业ESG话语的文本测度构建与漂绿识别研究"

# 在 github.com/new 手动创建空仓库 esg-greenwashing-nlp（不要勾选README）
# 然后关联远程并推送（替换 <你的用户名> 与 <你的PAT>）
git remote add origin https://github.com/<你的用户名>/esg-greenwashing-nlp.git
git push -u origin main
```

或使用 Token 免密推送：

```bash
git remote add origin https://<你的PAT>@github.com/<你的用户名>/esg-greenwashing-nlp.git
git push -u origin main
```

## 推荐仓库设置

- 仓库名：`esg-greenwashing-nlp`
- 描述：`企业ESG话语的文本测度构建与漂绿识别研究——NLP五步法全流程可复现（毕业论文模拟稿）`
- 可见性：Public（便于评审复现）或 Private（定稿前保密）
- Topics 建议：`nlp` `esg` `greenwashing` `text-mining` `thesis` `chinese-nlp`
