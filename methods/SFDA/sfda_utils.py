




def ema_update_CTRR(model,resnet_teacher,classifier_teacher,gamma):
    student = model.backbone.state_dict()
    teacher = resnet_teacher.state_dict()
    for key in student.keys():
        teacher[key].data.copy_(teacher[key].data * gamma +student[key].data * (1 - gamma))
        
    student = model.classifier.state_dict()
    teacher = classifier_teacher.state_dict()
    for key in student.keys():
        teacher[key].data.copy_(teacher[key].data * gamma +student[key].data * (1 - gamma))

    return resnet_teacher,classifier_teacher