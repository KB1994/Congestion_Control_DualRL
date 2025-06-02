from mininet.topo import Topo
from mininet.link import TCLink

class MyTopo(Topo):
    def build(self):
        # --------------------
        # Add switches
        # --------------------
        # Create four OpenFlow 1.3–enabled switches named s1–s4
        s1 = self.addSwitch('s1', protocols=["OpenFlow13"])
        s2 = self.addSwitch('s2', protocols=["OpenFlow13"])
        s3 = self.addSwitch('s3', protocols=["OpenFlow13"])
        s4 = self.addSwitch('s4', protocols=["OpenFlow13"])
        
        # --------------------
        # Add hosts
        # --------------------
        # Create six hosts (h1–h6), each with a fixed MAC address
        h1 = self.addHost('h1', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', mac='00:00:00:00:00:06')
        
        # --------------------
        # Host-to-switch links
        # --------------------
        # Connect h1, h2, h3 to switch s1 with 100 Mbps links
        self.addLink(h1, s1,
                     bw=100,                   # bandwidth in Mbps
                     intfName1='h1-eth0',      # host interface name
                     intfName2='s1-eth1')      # switch interface name
        self.addLink(h2, s1, bw=100, intfName1='h2-eth0', intfName2='s1-eth2')
        self.addLink(h3, s1, bw=100, intfName1='h3-eth0', intfName2='s1-eth3')
        
        # Connect h4, h5, h6 to switch s4 with 100 Mbps links
        self.addLink(h4, s4, bw=100, intfName1='h4-eth0', intfName2='s4-eth1')
        self.addLink(h5, s4, bw=100, intfName1='h5-eth0', intfName2='s4-eth2')
        # Note: fixed typo from 's4-et3' to 's4-eth3'
        self.addLink(h6, s4, bw=100, intfName1='h6-eth0', intfName2='s4-eth3')
        
        # --------------------
        # Switch-to-switch links
        # --------------------
        # s1 <-> s2: 10 Mbps link, HTB qdisc
        self.addLink(s1, s2,
                     bw=10,                    # bandwidth in Mbps
                     use_htb=True,             # enable hierarchical token bucket
                     intfName1='s1-eth4',
                     intfName2='s2-eth1')
        
        # s2 <-> s3: 1 Mbps link with small queue to create a bottleneck
        self.addLink(s2, s3,
                     bw=1,                     # very low bandwidth
                     max_queue_size=20,        # limit queue size to 20 packets
                     use_htb=True,
                     intfName1='s2-eth2',
                     intfName2='s3-eth1')
        
        # s3 <-> s4: 10 Mbps link, same as s1–s2
        self.addLink(s3, s4,
                     bw=10,
                     use_htb=True,
                     intfName1='s3-eth2',
                     intfName2='s4-eth4')

# Expose this topology as 'mytopo' so you can run: sudo mn --custom <thisfile>.py --topo mytopo
topos = {'mytopo': (lambda: MyTopo())}
