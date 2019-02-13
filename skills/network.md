### httpd service for mac
**Mac下启动Apache**  
```bash
# 1. start
sudo apachectl -k start
# 2. restart
sudo apachectl -k restart
# 设置Apache容器默认目录
cd /etc/apache2/
sudo vim httpd.conf  # 修改Apache配置文件
```
在vim中
```bash
/DocumentRoot # 查找 DocumentRoot 字符串。“/”为查找定位的意思

#将上述查找到的木木修改为自己想要的目录即可。
```
最后重启Apache即可

### 找到使用某个port的进程
```bash
lsof -i:<port>
```