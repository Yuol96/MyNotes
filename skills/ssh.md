# SSH Tunnel and Port Forwarding

## SSH 的 -L, -R, -D 三种用法
[Ref](http://blog.51cto.com/huanghai/1793850)

## How to ignore 跳板机?
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