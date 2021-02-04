---
title: UGR16 Dataset
author: 
---

<style>p {text-align: justify;}</style>

This page discusses the UGR16 IDS dataset created by Maciá-Fernández et al. This page is intended for learners seeking a dataset for a project or looking to learn about the UGR16 dataset.

<div style="text-align:center;"><img src="/assets/images/ugr16-dataset-network-flows.jpg" alt="." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">Network flows of the UGR16 dataset (Maciá-Fernández et al).</p>

## Dataset Summary

UGR16 was created to provide another choice for researchers testing Intrusion Detection Systems (IDS). The aim of the UGR16 creators was to solve the issue of representative data, UGR16 is designed to closely mimic a modern networks flows, with embedded synthetic cyber attacks to provide a useful standard dataset. The need arose as many researcher still use the DARPA '98/'99 datasets or derivatives of it. Networks and the devices attached to them have changed a great deal in 20 years.

The UGR16 dataset consits of real and synethetic <i>netflow</i> v9 data captured from sensors within a tier-3 ISP. The ISP is a cloud service provider, as such virtualized services are hosted including WordPress, Joomla, email, FTP etc. Victim machines were colocated alongside real clients and attacker machines placed outside the network (see Network Typology section below). Attacks are generated at fix and random times, allowing anomoly detectors to be assessed. Real botnet traffic captured from the malware <i>Neris</i> was inserted into the network data during capture.

The dataset contains four months worth of network flows captured, typically network capture datasets only cover a few days. This long duration allows the UGR16 dataset to capture neuances between day-night, day-to-day, weekday-to-weekend and week-to-week traffic, resulting in a dataset which is more representative of a real large network. Th large capture results in over 600 million external IPs observed, with 10 million corresponding subdomains and 16 billion individual flows captued over the time frame. The researchers note several observations; a surprising number of ISP clients use Telnet to manage services, P2P traffic is low, peaks in STMP traffic exist - attributed to both legitimate and spam email campaigns, and the constant SMB traffic is attributed to a single retail company. 

## UGR16 In Detail

### Network Typology

The UGR dataset was collected from an tier-3 ISP cloud server provider. Below we describe the network typology from which the researchers captured their data.

<div style="text-align:center;"><img src="/assets/images/ugr16-dataset-network-typology.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Network typology of UGR16 dataset capture (Maciá-Fernández et al).</p>

* <i>BR1</i> and <i>BR1</i> are redundant routers connected to the Internet. Data is capture from three sensors connected to these routers.
* The ISP has two sub-networks; <i>core</i> and <i>inner</i>. <i>Core</i> services are not protected by a firewall, whereas <i>Inner</i> services which belong to the clients are protected.
* Maciá-Fernández et al configure 5 attacker machines external to the network, termed <i>A<sub>1</sub>-A<sub>5</sub></i>.
* 5 victim machines are configured within the <i>core</i>, colocated with real clients in the network <i>V<sub>1</sub></i>.
* 15 additional victim machines are configured within the <i>inner</i> network, which is split into 3 networks, with 5 machines in each. The networks are labelled as network <i>V<sub>2</sub>-V<sub>4</sub></i>, with corresponding victim machine names; the 1st machine in network <i>V<sub>2</sub></i> is <i>V<sub>21</sub></i>, the second victim machine in network <i>V<sub>3</sub></i> is <i>V<sub>32</sub></i>.

### Attack Generation

Attack data was simulated from the attacker machines against the victim machines as described in the section above. Synthetic attack data can be a problem if it is not created in a realistic manner, particularly in terms of attack identifiation. Non-realistic behaviour will be analysed by a machine learning algorithm and may result in a useless model which has learned unrealistic patterns. However, capturing real attack traffic leads to even more difficulties. Understanding when attacks occured, against whom and from where are difficult to derive unless they are created rather than observed. Creating simulated traffic allows researchers the ability identify attacks with certainty and benchmark machine learning algorithms more accurately.

Maciá-Fernández et al simulate network-related attacks exclusively, as ultimately the researchers only capture netflow traffic and not payload information (which would be included in the pcap format). They focus on three broad types of attack which are generated every 2 hours; low-rate DoS, port scanning and bot traffic.

* **Low-rate DoS**. 1280 bit packets are sent at a rate of 100 packets per second to port 80 of each vitcim machine incorperated into three attack scenarios.
    * <i>DoS11</i>: 1 vs 1, an attack machine attacks one victim machine. The duration of the attack is 3 minutes.
    * <i>DoS53s</i>: A simultanious attack against 3 victim machines by all 5 attacker machines. The attack duration is 3 minutes.
    * <i>DoS53a</i>: An asynchronous, sequential attack against 3 victim machines by all 5 attacker machines. The duration is 3 minutes, with a 30 second period of inactivity between attacks.
* **Port scanning**. A continuious SYN scan of common ports for 3 minutes.
    * <i>Scan11</i>: 1 vs 1 scan, duration 3 minutes.
    * <i>Scan44</i>: 4 vs 4 scan, simultanious with a duration of 3 minutes.
* **Bot traffic**. Real recorded botnet traffic is inserted into the data, mimicing a real infection. The network could not really be infected due to ethical reasons. The botnet traffic is part of the CTU-13 dataset based of an infection of the malware <i>Neris</i>.

Attacks are generated in batches, scheduled every two hours in one of two patterns; <i>scheduled</i> or <i>random</i>. <i>Scheduled</i> attacks are executed at fixed and known time intervals from an offset based on the time the attack generation started, e.g. +01h03m. <i>Random</i> attacks are executed at random intervals between start time +00h00m and +01h50m. Overlaps can exist within attacks, and botnet traffic is not included. 

## Attack Labeling

Maciá-Fernández et al label the UGR16 dataset using two methods; <i>signitue-based labeling</i> and <i>anomoly-based labelling</i>.

* <i>Signiture-based labeling</i>: when possible, attacks were labelled using their signiture (including attacks generated by the researchers) using data from <i>hpHosts</i><a href="#footnote">†</a>, <i>Malware DL</i>, <i>Spamhaus</i> and <i>Abuse.ch</i>. Attacks with the same signiture have been grouped together, for example the generated DoS attacks we discussed above are labelled "DoS".
* <i>Anomoly-based labeling</i>: three 'state-of-the-art' anomaly detection were used to identify attacks within the background traffic (see section 5.5.2, p. 420 (p. 10 in PDF)). The researchers identify attacks where the three detectors reach a consensus and manual analysis was used to create a signiture for the attack. Three attacks were discovered:
    * <i>UDP scans</i> were identified and labeled "anomaly-udpscan". Researchers attributed this to malware activity from a German IP address.
    * <i>SSH scans</i> were identified and labeled "anomaly-sshscan". A machine within the ISP cloud was conducting a large number of SSH scans against targets in South America.
    * <i>Email spam</i> was identified and labeled "anomaly-spam". A spike in SMTP traffic was observed, 5 IPs generated a total of 12.5 million STMP connectons. The IP addresses generating the traffic ended up on Yahoo's blacklist.

<div style="text-align:center;"><img src="/assets/images/ugr16-dataset-distribution-anomaly-.png" alt="." style="width:650px;"/></div>

<p style="text-align: center; font-style: italic;">Traffic duration visualized for UDP scans detected (Maciá-Fernández et al).</p>

## UGR16 vs Other Datasets

UGR16 was created to overcome some of the specific limitations of other IDS datasets:

* entirely / high-ratio of synthetic data
* short capture duration
* non-representitive of day/night cycle

<div style="text-align:center;"><img src="/assets/images/ugr16-dataset-versus-other-datasets-table.png" alt="." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">.</p>

## Summary

The UGR16 dataset is unusual in its long duration of capture - 4 months - few other network flow datasets are this long and contain data captured from a large ISP. The methodolgy of capture and attack generation allow the dataset to contain labelled attack traffic, while retaining the realism of real traffic and avoiding the pitfalls of dataset consisting entirely of synthetic data.

<h5 id="footnote">Footnote</h5>

hpHost blocklist's are no longer availalbe and the website now redirects to Malwarebytes. <a href="https://www.reddit.com/r/pfBlockerNG/comments/ft243q/hphosts_is_gone_redirects_to_malwarebytescom/" target="_blank">Link</a> to reddit thread discussing the issue.

---

## References

<ul>

<li>Garcia, S., Grill, M., Stiborek J., and Zunino, A. (2014) <i>The CTU-13 Dataset. A Labeled Dataset with Botnet, Normal and Background traffic.</i>. <a href="https://www.stratosphereips.org/datasets-ctu13" target="_blank">https://www.stratosphereips.org/datasets-ctu13</a></li>

<li>Maciá-Fernández, G., Camacho, J., Magán-Carrión, R., García-Teodoro, P., and Theron, R. (2016) <i>UGR'16: A New Dataset for the Evaluation of Cyclostationarity-Based Network IDSs</i>. <a href="https://nesg.ugr.es/nesg-ugr16/" target="_blank">https://nesg.ugr.es/nesg-ugr16/</a></li>

<li>Spamhaus (2020) <i>Spamhaus</i>. <a href="https://www.spamhaus.org/" target="_blank">https://www.spamhaus.org/</a> </li>

<li>Abuse.ch (2020) <i>Abuse.ch Fighting Malware and Botnets</i>. <a href="https://abuse.ch/" target="_blank">https://abuse.ch/</a> </li>

<li>hpHosts via Internet Archive (2020) <i>hpHosts Online</i>. <a href="https://web.archive.org/web/20200211101232/https://hosts-file.net/" target="_blank">https://web.archive.org/web/20200211101232/https://hosts-file.net/</a> </li>

<li>Malware DL (2020) <i>Malware Domain List</i>. <a href="https://www.malwaredomainlist.com/" target="_blank">https://www.malwaredomainlist.com/</a> </li>

</ul>
