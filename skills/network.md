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

### build a file server
#### Node.js http-server
```bash
npm install http-server
./node_modules/http-server/bin/http-server <path-to-target-directory>

npm install http-server -g
http-server <path-to-target-directory>
http-server --help
```

#### Apache httpd
[Mac解决Apache2目录权限问题](https://www.cnblogs.com/sweetheartly/articles/9439858.html)  
[Mac下如何修改apache根目录](https://www.cnblogs.com/chuangshaogreat/p/7821407.html)  
[MAC自带apache开启目录访问及基本配置](https://www.jianshu.com/p/a617691a7f4f)

#### browser-sync
```bash
# move to the target directory
browser-sync start --server --directory --files "**/*" --browser
```

#### ftpd-cli
[Truly simple zero-config 5-sec FTP server in pure JavaScript](https://www.npmjs.com/package/ftpd-cli)  
```bash
npm install -g ftpd-cli
# To start ftp daemon with anonymous access:
ftpd-cli --directory path/to/files
# To start ftp daemon with login-passwords only:
ftpd-cli --directory path/to/files --user-password login:322
# To stop daemon:
ftpd-cli --stop
```

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


 