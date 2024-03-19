all: sender receiver

sender:
	gcc -m32 sender.c -o sender

receiver:
	gcc -m32 receiver.c -D__DEBUG__ -o receiver
#	gcc -m32 receiver.c -o receiver

clean:
	rm -f receiver sender
