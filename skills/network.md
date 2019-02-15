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

### Ubuntu 搭建Http代理服务器
1. shadowsocks
```bash
pip3 install shadowsocks
ssserver -p 8388 -k <password> -m rc4-md5 --user <username> -d start  
```
2. privoxy
```bash
apt-get install privoxy
vim /etc/privoxy/config  
```
在config中修改属性：
```vim
listen-address  :8118
enable-remote-toggle  1
```
在文本末尾添加：
```vim
forward-socks5 / 127.0.0.1:8388  .
```
最后重启privoxy
```bash
service privoxy restart  
```
3. 停止代理
```bash
 service privoxy stop
 ```


 