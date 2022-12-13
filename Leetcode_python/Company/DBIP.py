class Solution:
    def canPass(self, firewall: dict, ipStr: str) -> bool:
        allowRule = set()
        denyRule = set()

        def transferIP(ip : str):
            mask_list = ip.split(".")
            # rule_str = []
            # for t in mask_list:
            #     temp_binary = bin(int(t)).replace('0b', '')
            #     pad_size = 8 - len(temp_binary)
            #     rule_str.append("0" * pad_size + temp_binary)  
            # return "".join(rule_str)
            rule_32 = int(mask_list[0])
            for i in range(1, len(mask_list)):
                rule_32 = rule_32 << 8
                rule_32 += int(mask_list[i])
            return rule_32


        for k, v in firewall.items():
            rule = k
            type = v
            sep = rule.find("/")
            mask_digit = int(rule[sep+1:])
            mask_ip = rule[0:sep]
            rule_str = transferIP(mask_ip)
            if type == "allow":
                allowRule.add((rule_str, mask_digit))
            else :
                denyRule.add((rule_str, mask_digit))

        targetStr = transferIP(ipStr)
        
        print(allowRule)
        print(denyRule)
        for r in denyRule:
            content = r[0]
            # digit = r[1]
            digit = 32 - r[1]
            # if content[:digit] == targetStr[:digit]:
            if (content << digit) == (targetStr << digit):
                return False
        
        for r in allowRule:
            content = r[0]
            # digit = r[1]
            digit = 32 - r[1]
            # if content[:digit] == targetStr[:digit]:
            if (content << digit) == (targetStr << digit):
                return True
        
        return False

rule = {"192.168.1.22/24":"allow", "192.168.122.126/20":"deny"}
obj = Solution()
ip = "192.168.122.22"
print(obj.canPass(rule, ip))
