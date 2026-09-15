# Test Queries and System Outputs

10 representative queries run against the live system (`uv run python -m scripts.run_sample_queries`), covering 9 in-scope questions across three difficulty tiers plus one out-of-scope question.

*Results may differ depending on the backend model and generated output tokens.*

## Easy — single-subject direct lookup

### 1. What is a potential indicator of the 'T1539: Steal Web Session Cookie' attack technique?

**Answer:**

Potential indicators of the 'T1539: Steal Web Session Cookie' attack technique include:

*   **Process Access:** Attempts by programs to dump or inject into browser process memory [1].
*   **File Access:** User attempts to gain access to a computing resource or network, specifically by providing credentials to cloud service management consoles (some cloud providers, like AWS, provide distinct log events for these login attempts) [2].

**Sources:**

- `relationships_detects/relationships_detects__T1539__001.txt`
- `relationships_detects/relationships_detects__T1539__002.txt`

### 2. Which mitigations are recommended for 'T1110: Brute Force'?

**Answer:**

The recommended mitigation strategies for 'T1110: Brute Force' are:
* M1036: Account Use Policies [1]
* M1032: Multi-factor Authentication [1]
* M1027: Password Policies [1]
* M1018: User Account Management [1]

**Sources:**

- `relationships_mitigations_summaries/relationships_mitigations_summaries__T1110__001.txt`

### 3. What data sources can be used to detect 'T1055: Process Injection'?

**Answer:**

To detect 'T1055: Process Injection', the following 7 data components can be used:
* File Metadata [1]
* File Modification [1]
* Module Load [1]
* OS API Execution [1]
* Process Access [1]
* Process Metadata [1]
* Process Modification [1]

**Sources:**

- `relationships_detections_summaries/relationships_detections_summaries__T1055__001.txt`

## Medium — single-subject synthesis or attribution

### 4. What software has the threat group APT29 been observed using?

**Answer:**

APT29 has been observed using the following software:

*   **CosmicDuke** [1]
*   **Impacket** [2]
*   **SeaDuke** [3]
*   **GeminiDuke** [4]
*   **HAMMERTOSS** [5]
*   **PowerDuke** [6]
*   **FatDuke** [7]
*   **TrailBlazer** [8]
*   **Raindrop** [9]
*   **SUNBURST** [10]
*   **SUNSPOT** [11]
*   **TEARDROP** [12]

Additionally, APT29 conducted the SolarWinds Compromise campaign [13], which utilized the following software:
*   **Mimikatz** [14][15]
*   **Cobalt Strike** [15]
*   **AdFind** [15]
*   **SUNBURST** [15]
*   **TEARDROP** [15]
*   **SUNSPOT** [15]
*   **Raindrop** [15]
*   **GoldMax** [15]
*   **Sibot** [15]
*   **GoldFinder** [15][16][17]
*   **TrailBlazer** [15]

**Sources:**

- `relationships_groups_for_software/relationships_groups_for_software__S0050__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0357__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0053__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0049__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0037__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0139__001.txt`
- `relationships_groups_for_software/relationships_groups_for_software__S0512__001.txt`
- `software/software__S0682__001.txt`
- `software/software__S0565__001.txt`
- `software/software__S0559__001.txt`
- `software/software__S0562__001.txt`
- `software/software__S0560__001.txt`
- `campaigns/campaigns__C0024__001.txt`
- `relationships_campaigns_for_software/relationships_campaigns_for_software__S0002__001.txt`
- `relationships_software_for_campaign/relationships_software_for_campaign__C0024__001.txt`
- `software/software__S0597__001.txt`
- `relationships_campaigns_for_software/relationships_campaigns_for_software__S0597__001.txt`

### 5. How can Process Creation be used to detect Remote System Discovery?

**Answer:**

To detect the attack technique 'T1018: Remote System Discovery' using the 'Process Creation' data component, monitor for newly executed processes that can be used to discover remote systems, such as tracert.exe and ping.exe, particularly when these processes are executed in quick succession [1].

**Sources:**

- `relationships_detects/relationships_detects__T1018__002.txt`

### 6. What is the CVE number for the Log4Shell vulnerability?

**Message:**

The knowledge base doesn't contain enough evidence to answer this. The provided excerpts list various CVEs associated with different attack software, but none of them mention 'Log4Shell' or its corresponding CVE (CVE-2021-44228).

**Sources:**

- `relationships_uses_software/relationships_uses_software__T1190__002.txt`
- `relationships_uses_software/relationships_uses_software__T1068__006.txt`
- `relationships_uses_software/relationships_uses_software__T1203__002.txt`
- `relationships_uses_software/relationships_uses_software__T1203__004.txt`
- `relationships_uses_software/relationships_uses_software__T1203__009.txt`
- `relationships_uses_software/relationships_uses_software__T1068__003.txt`
- `relationships_uses_software/relationships_uses_software__T1210__001.txt`
- `relationships_uses_software/relationships_uses_software__T1203__005.txt`
- `relationships_uses_software/relationships_uses_software__T1210__002.txt`
- `relationships_uses_software/relationships_uses_software__T1091__016.txt`
- `relationships_uses_software/relationships_uses_software__T1068__015.txt`
- `relationships_uses_software/relationships_uses_software__T1068__005.txt`
- `relationships_uses_software/relationships_uses_software__T1068__004.txt`

## Hard — multi-subject comparison or broad synthesis

### 7. Compare the detection methods for T1110 Brute Force and T1556 Modify Authentication Process.

**Answer:**

The detection methods for T1110 Brute Force and T1556 Modify Authentication Process are as follows:

**T1110 Brute Force**
Detection relies on three data components [1]:
*   **Application Log Content:** Monitoring authentication logs for high volumes of system and application login failures of Valid Accounts [2].
*   **User Account Authentication:** Monitoring for numerous failed authentication attempts across various accounts, which may indicate password spraying [3].
*   **Command Execution:** Monitoring executed commands and arguments that may utilize brute force techniques to gain account access when passwords are unknown or hashes are obtained [4].

**T1556 Modify Authentication Process**
Detection involves 13 data components [5]: Active Directory Object Modification, Application Log Content, Cloud Service Modification, File Creation, File Modification, Logon Session Creation, Module Load, OS API Execution, Process Access, User Account Authentication, User Account Modification, Windows Registry Key Creation, and Windows Registry Key Modification. 

Specific detection methods include:
*   **File Modification:** Monitoring for suspicious modifications to files associated with authentication processes (such as module paths and configuration files like `/etc/pam.d/`) using monitoring tools like auditd or system-integrity tools like AIDE, as well as monitoring access to cryptographic keys and certificates [6].

**Sources:**

- `relationships_detections_summaries/relationships_detections_summaries__T1110__001.txt`
- `relationships_detects/relationships_detects__T1110__003.txt`
- `relationships_detects/relationships_detects__T1110__001.txt`
- `relationships_detects/relationships_detects__T1110__002.txt`
- `relationships_detections_summaries/relationships_detections_summaries__T1556__001.txt`
- `relationships_detects/relationships_detects__T1556__007.txt`

### 8. Which threat groups have used both Mimikatz and Cobalt Strike?

**Answer:**

The threat groups that have used both Mimikatz (S0002) and Cobalt Strike (S0154) include:

*   DarkHydrus (G0079) [1]
*   Cobalt Group (G0080) [2]
*   Threat Group-3390 (G0027) [3]
*   Chimera (G0114) [4]
*   Earth Lusca (G1006) [5]
*   Indrik Spider (G0119) [6]
*   TA505 (G0092) [7]
*   FIN6 (G0037) [8]
*   APT32 (G0050) [9]

**Sources:**

- `relationships_software_for_group/relationships_software_for_group__G0079__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0080__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0027__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0114__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G1006__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0119__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0092__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0037__001.txt`
- `relationships_software_for_group/relationships_software_for_group__G0050__001.txt`

### 9. How do I detect lateral movement?

**Answer:**

To detect lateral movement, you can monitor the following data components and activities:

**Network Traffic and Connections**
*   **Network Traffic Flow:** Monitor for uncommon data flows, specifically processes that do not normally communicate over the network or have never been seen before [1]. In the case of host, port, or service scans, a single machine will make multiple calls to other hosts to identify live services [1]. For SMB/Windows Admin Shares, monitor for uncommon SMB data flows [2] and specifically SMB write requests (e.g., `smb2_write_request`), which are often used to move laterally [2][3].
*   **Network Connection Creation:** Monitor for new network connections associated with pings or scans used to list other systems by hostname, IP address, or other logical identifiers [4].

**Logon and Authentication**
*   **Logon Session Metadata:** Analyze authentication logs for unusual RDP access patterns by comparing remote desktop logons against known/approved originating systems [5].
*   **Logon Session Creation:** 
    *   Monitor for logon behavior (e.g., Event ID 4624 Logon Type 3) using valid accounts to interact with remote network shares via SMB [6].
    *   Review newly created logons for discrepancies; unusual remote logins that correlate with other suspicious activity, such as executing binaries, may indicate malicious activity [7].
*   **User Account Authentication:** To detect "Pass the Hash," monitor for NTLM LogonType 3 authentications that are not anonymous logins and are not associated with a domain login [8].
*   **Active Directory Credential Requests:** Monitor for requests for new service tickets or ticket granting tickets (TGT) to a Domain Controller (e.g., Windows Security events 4768 and 4769) [9].

**Process and Command Execution**
*   **Process Creation:** 
    *   Monitor for newly executed processes that gather information on domain trust relationships [10] or look for folders and drives shared on remote systems [11].
    *   Monitor for suspicious processes spawned in a short period that perform Discovery or gather system information [12].
*   **Command Execution:** Monitor executed commands and arguments that attempt to list other systems by logical identifier, IP, or hostname [13][14]. For network devices, monitor AAA logs for commands run by unauthorized or unexpected users [14].
*   **OS API Execution:** Monitor for API calls associated with gathering domain trust relationships (e.g., `DSEnumerateDomainTrusts()`) [15] or those looking for shared drives and folders on remote systems [16].

**Other Indicators**
*   **PowerShell Logs:** Event ID 4104 can capture script block contents used to enumerate systems with RDP access [14].
*   **.NET Methods:** The use of `GetAllTrustRelationships()` can be an indicator of Domain Trust Discovery [15].

**Sources:**

- `relationships_detects/relationships_detects__T1046__002.txt`
- `relationships_detects/relationships_detects__T1021.002__002.txt`
- `relationships_detects/relationships_detects__T1021.002__005.txt`
- `relationships_detects/relationships_detects__T1018__004.txt`
- `relationships_detects/relationships_detects__T1021.001__004.txt`
- `relationships_detects/relationships_detects__T1021.002__004.txt`
- `relationships_detects/relationships_detects__T1550.002__002.txt`
- `relationships_detects/relationships_detects__T1550.002__001.txt`
- `relationships_detects/relationships_detects__T1550.002__003.txt`
- `relationships_detects/relationships_detects__T1482__001.txt`
- `relationships_detects/relationships_detects__T1135__003.txt`
- `relationships_detects/relationships_detects__T1497.002__003.txt`
- `relationships_detects/relationships_detects__T1135__001.txt`
- `relationships_detects/relationships_detects__T1018__001.txt`
- `relationships_detects/relationships_detects__T1482__002.txt`
- `relationships_detects/relationships_detects__T1135__002.txt`

## Out-of-scope — refused by the Scope Gate

### 10. What's a good recipe for chocolate chip cookies?

**Message:**

This question isn't about cybersecurity, so it's out of scope for this system.
