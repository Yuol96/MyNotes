### SSH 的 -L, -R, -D 三种用法
[Ref](http://blog.51cto.com/huanghai/1793850)

### How to ignore 跳板机?
Take `rsync` as an example:

```bash
#!/bin/sh
MAPPED=`sockstat -4l|grep :10022`

if [ "X$MAPPED" = X ] ; then

echo "Frist ,we need map remote ssh port to local via ssh,input password:" 
#跳板机passwd
ssh -fN -L 10022:<server_ip>:<server_port> -p <跳板ssh port> <跳板user>@<跳板ip>
fi

echo "Start sync ...."

rsync -ave "ssh -p 10022" /local-data-path/ <server_user>@127.0.0.1:/remote-data-path/ 

# 需要输入服务器密码
```

### 在mac OS X上启动sshd服务
直接运行`sshd`会报错：
```bash
sshd 
# sshd re-exec requires execution with an absolute path
```
也就是说需要使用sshd这个命令的绝对路径：
```bash
/usr/sbin/sshd
# Could not load host key: /etc/ssh/ssh_host_rsa_key
# Could not load host key: /etc/ssh/ssh_host_dsa_key
# Could not load host key: /etc/ssh/ssh_host_ecdsa_key
# Could not load host key: /etc/ssh/ssh_host_ed25519_key
# sshd: no hostkeys available -- exiting.
```
现在报错原因是，在 SSH 连接协议中需要有 RSA 或 DSA 密钥的鉴权。 因此，我们可以在服务器端使用 ssh-keygen 程序来生成一对公钥/私钥对
```bash
ssh-keygen -t rsa -b 2048 -f /etc/ssh/ssh_host_rsa_key
```
上面 ssh-keygen 命令中，
-t 选项表示生成的密钥所使用的加密类型，这里选择的是 RSA ；
-b 选项表示 bit，后接一个整数，表示加密的位数，该数值越大表示加密的强度越高；
-f 选项后接要生成的密钥文件名。根据 /etc/ssh 目录下的 sshd_config 配置文件，RSA 密钥默认识别文件名为 ssh_host_rsa_key 。
命令执行成功后，在 /etc/ssh 下会看到有两个文件生成：ssh_host_rsa_key 和 ssh_host_rsa_key.pub ，前者是私钥，后者是公钥。

还有一种方式是直接使用`~/.ssh/id_rsa`作为密钥文件：
```bash
/usr/sbin/sshd -h ~/.ssh/id_rsa
```
命令执行后没有任何输出。可以通过`ps aux | egrep sshd` 来查看进程

### 保持ssh的长连接
**有以下三种方法**

1. 在server端修改 `/etc/ssh/sshd_config`

```vim
ClientAliveInterval 60 ＃server每隔60秒发送一次请求给client，然后client响应，从而保持连接
ClientAliveCountMax 3 ＃server发出请求后，客户端没有响应得次数达到3，就自动断开连接，正常情况下，client不会不响应
```

2. 在client端修改 `/etc/ssh/ssh_config`

```vim
ServerAliveInterval 60 ＃client每隔60秒发送一次请求给server，然后server响应，从而保持连接
ServerAliveCountMax 3  ＃client发出请求后，服务器端没有响应得次数达到3，就自动断开连接，正常情况下，server不会不响应
```

3. 在命令行参数中加入新的option

```bash
ssh -o ServerAliveInterval=60 xxxxxxx
```
这样子只会在需要的连接中保持持久连接， 毕竟不是所有连接都要保持持久的。
