"""구조 격자의 사면체 연결과 바깥 방향 경계면을 생성해요."""
from itertools import permutations
import numpy as np

def make_block(width=2., depth=1., height=.5, nx=3, ny=3, nz=3):
    """직육면체의 절점과 사면체 및 경계면을 일정한 순서로 생성해요."""
    sizes = (width, depth, height)
    counts = (nx, ny, nz)
    if any(not np.isfinite(s) or s <= 0 for s in sizes):
        raise ValueError('dimensions must be finite and positive')
    if any(not isinstance(n, (int, np.integer)) or n < 2 for n in counts):
        raise ValueError('nx, ny, nz must be integers >= 2')
    z, y, x = np.meshgrid(np.linspace(0., height, nz), np.linspace(0., depth, ny),
                           np.linspace(0., width, nx), indexing='ij')
    nodes = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    offsets = []
    strides = np.array([1, nx, nx * ny])
    for order in permutations(range(3)):
        corners = np.zeros((4, 3), dtype=int)
        for step, axis in enumerate(order, start=1):
            corners[step] = corners[step - 1]
            corners[step, axis] += 1
        if np.linalg.det(corners[1:]) < 0:
            corners[[1, 2]] = corners[[2, 1]]
        offsets.append(corners @ strides)
    k, j, i = np.meshgrid(np.arange(nz - 1), np.arange(ny - 1),
                          np.arange(nx - 1), indexing='ij')
    # 격자 셀의 시작 절점 번호를 구해요: $$b=i+n_x(j+n_yk)$$
    bases = (i + nx * (j + ny * k)).ravel()
    tets = (bases[:, None, None] + np.asarray(offsets)[None, :, :]).reshape(-1, 4)
    face_order = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = tets[:, face_order].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                  return_index=True, return_counts=True)
    boundary = faces[np.sort(first[counts == 1])]
    top = boundary[np.all(np.isclose(nodes[boundary, 2], height), axis=1)]
    bottom = np.flatnonzero(np.isclose(nodes[:, 2], 0.))
    return dict(nodes=nodes, tetrahedra=tets, boundary_faces=boundary,
                top_faces=top, bottom_nodes=bottom)

def make_spherical_cap(radius=10., half_width=6., height=10., nxy=17, nz=9,
                       clearance=.002, grading=5., depth_grading=3.):
    r"""구면 하부와 평평한 윗면을 갖는 등급 격자를 생성해요.

    $$z=c+R-\sqrt{R^2-x^2-y^2}+(h-R+\sqrt{R^2-x^2-y^2})\eta$$
    """
    values=(radius, half_width, height, clearance, grading, depth_grading)
    if not all(np.isfinite(x) for x in values):
        raise ValueError('cap parameters must be finite')
    if half_width<=0 or radius<=np.sqrt(2.)*half_width:
        raise ValueError('require radius > sqrt(2)*half_width > 0')
    if clearance<=0 or not 0<=grading<=10 or not 1<=depth_grading<=6:
        raise ValueError('require clearance>0, grading in [0,10], depth_grading in [1,6]')
    if not isinstance(nxy,(int,np.integer)) or nxy<5 or nxy%2!=1:
        raise ValueError('nxy must be an odd integer >= 5')
    # 모서리에서 구면 높이를 계산해요: $$z_e=R-\sqrt{R^2-2a^2}$$
    edge_profile=radius-np.sqrt(radius*radius-2*half_width*half_width)
    if height<=edge_profile:
        raise ValueError('flat top must be above the curved lower surface')
    mesh=make_block(width=2.,depth=2.,height=1.,nx=nxy,ny=nxy,nz=nz)
    local=mesh['nodes'];bottom=mesh['bottom_nodes']
    mesh['contact_faces']=mesh['boundary_faces'][
        np.all(np.isin(mesh['boundary_faces'],bottom),axis=1)]
    def graded(t):
        r"""중앙에 절점을 집중시키는 좌표 변환을 적용해요.

        $$x=a\sinh(gt)/\sinh(g)$$
        """
        # 중앙을 촘촘하게 하는 좌표를 구해요: $$x=a\sinh(gt)/\sinh(g)$$
        return half_width*(np.sinh(grading*t)/np.sinh(grading) if grading else t)
    x,y=graded(local[:,0]-1.),graded(local[:,1]-1.)
    # 구면 하부 높이를 계산해요: $$z_b=R-\sqrt{R^2-x^2-y^2}$$
    profile=radius-np.sqrt(radius*radius-x*x-y*y)
    # 깊이 방향의 등급 좌표를 구해요: $$\eta=t^d$$
    eta=local[:,2]**depth_grading
    # 구면과 평면 사이의 절점 좌표를 계산해요: $$z=c+z_b+(h-z_b)\eta$$
    mesh['nodes']=np.column_stack((x,y,clearance+profile+(height-profile)*eta))
    return mesh



def make_cylinder(radius=3., half_length=3., n_angle=32, n_radial=1, n_axis=7,
                  facing='up', angle_degrees=0., clearance=0.,
                  query_half_width=.4, patch_half_width=1.2):
    r"""완전한 원통을 일정한 순서의 사면체와 바깥 방향 경계면으로 나눠요.

    $$\boldsymbol{x}_{kij}=R_z(\alpha)(r_i\sin\theta_j,\ y_k,\ z_c+\sigma r_i\cos\theta_j)^T$$

    양 끝 원판을 고정 영역으로 두고, 접촉 질의면보다 넓은 면에서 법선을 계산해요.
    """
    values=(radius,half_length,angle_degrees,clearance,query_half_width,patch_half_width)
    if not all(np.isfinite(v) for v in values) or min(radius,half_length,query_half_width)<=0 or clearance<0:
        raise ValueError('finite positive cylinder dimensions required')
    if not query_half_width<patch_half_width<min(radius,half_length):
        raise ValueError('require 0 < query_half_width < patch_half_width < radius, half_length')
    if facing not in ('up','down'):raise ValueError('facing must be up or down')
    if not isinstance(n_angle,(int,np.integer)) or n_angle<16 or n_angle%8:
        raise ValueError('n_angle must be divisible by 8 and >= 16')
    if not isinstance(n_radial,(int,np.integer)) or n_radial<1:
        raise ValueError('n_radial must be an integer >= 1')
    if not isinstance(n_axis,(int,np.integer)) or n_axis<7 or n_axis%2!=1:
        raise ValueError('n_axis must be odd and >= 7')
    m=n_angle//4
    # 사분면 안의 균등 매개변수를 계산해요: $$t_j=j/m$$
    t=np.arange(m)/m
    # 접촉 극점 주변의 각도를 촘촘하게 배치해요: $$\theta_{qj}=q\pi/2+(\pi/4)(1-\cos(\pi t_j))$$
    theta=np.concatenate([q*np.pi/2+(np.pi/4)*(1-np.cos(np.pi*t)) for q in range(4)])
    disk=[(0.,0.)]
    for ring in range(1,n_radial+1):
        # 내부 원환의 반지름을 계산해요: $$r_i=Ri/n_r$$
        r=radius*ring/n_radial
        # 원판의 수평 좌표를 계산해요: $$x_{ij}=r_i\sin\theta_j$$
        disk_x=r*np.sin(theta)
        # 원판의 수직 좌표를 계산해요: $$z_{ij}=r_i\cos\theta_j$$
        disk_z=r*np.cos(theta)
        disk.extend(zip(disk_x,disk_z))
    disk=np.asarray(disk)
    width=len(disk)
    j=np.arange(n_angle)
    k=(j+1)%n_angle
    triangles=[np.column_stack((np.zeros(n_angle,dtype=int),1+j,1+k))]
    for ring in range(1,n_radial):
        inner=1+(ring-1)*n_angle
        outer=inner+n_angle
        pair=np.empty((n_angle,2,3),dtype=int)
        pair[:,0]=np.column_stack((inner+j,outer+j,outer+k))
        pair[:,1]=np.column_stack((inner+j,outer+k,inner+k))
        triangles.append(pair.reshape(-1,3))
    triangles=np.concatenate(triangles)
    # 축 방향의 절점을 중앙에 집중시켜요: $$y_k=L\sinh(3t_k)/\sinh(3)$$
    axis=half_length*np.sinh(3*np.linspace(-1.,1.,n_axis))/np.sinh(3.)
    local=np.empty((n_axis*width,3))
    local[:,0]=np.tile(disk[:,0],n_axis)
    local[:,1]=np.repeat(axis,width)
    local[:,2]=np.tile(disk[:,1],n_axis)
    # 마주 보는 면의 높이와 초기 간격을 적용해요: $$z=z_d-R\ (\sigma=1),\qquad z=c+R-z_d\ (\sigma=-1)$$
    local[:,2]=(local[:,2]-radius if facing=='up' else clearance+radius-local[:,2])
    # 원통 축의 회전각을 라디안으로 바꿔요: $$\alpha=\pi\alpha_{\mathrm{deg}}/180$$
    a=np.deg2rad(angle_degrees)
    nodes=local.copy()
    # 회전한 첫 좌표를 계산해요: $$x'=\cos\alpha\,x-\sin\alpha\,y$$
    nodes[:,0]=np.cos(a)*local[:,0]-np.sin(a)*local[:,1]
    # 회전한 둘째 좌표를 계산해요: $$y'=\sin\alpha\,x+\cos\alpha\,y$$
    nodes[:,1]=np.sin(a)*local[:,0]+np.cos(a)*local[:,1]
    prism=np.sort(triangles,axis=1)[None,:,:]+width*np.arange(n_axis-1)[:,None,None]
    tets=np.empty((n_axis-1,len(triangles),3,4),dtype=int)
    tets[:,:,0,:3]=prism
    tets[:,:,0,3]=prism[:,:,2]+width
    tets[:,:,1,:2]=prism[:,:,:2]
    tets[:,:,1,2:]=prism[:,:,1:]+width
    tets[:,:,2,0]=prism[:,:,0]
    tets[:,:,2,1:]=prism+width
    tets=tets.reshape(-1,4)
    xyz=nodes[tets]
    # 사면체의 방향과 체적 부호를 계산해요: $$6V_e=\det(\boldsymbol{x}_1-\boldsymbol{x}_0,\boldsymbol{x}_2-\boldsymbol{x}_0,\boldsymbol{x}_3-\boldsymbol{x}_0)$$
    det=np.linalg.det(xyz[:,1:]-xyz[:,:1])
    bad=np.flatnonzero(det<0)
    tets[bad,1],tets[bad,2]=tets[bad,2].copy(),tets[bad,1].copy()
    if np.any(det==0):raise ValueError('degenerate cylinder tetrahedron')
    face_order=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces=tets[:,face_order].reshape(-1,3)
    _,first,counts=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    boundary=faces[np.sort(first[counts==1])]
    end_mask=np.all(np.isclose(local[boundary,1],-half_length),axis=1)|np.all(np.isclose(local[boundary,1],half_length),axis=1)
    side=boundary[~end_mask]
    # 경계 삼각형의 방향을 계산해요: $$\boldsymbol{a}_f=(\boldsymbol{x}_1-\boldsymbol{x}_0)\times(\boldsymbol{x}_2-\boldsymbol{x}_0)$$
    normal=np.cross(nodes[side[:,1]]-nodes[side[:,0]],nodes[side[:,2]]-nodes[side[:,0]])
    # 상대 원통을 향하는 면을 선택해요: $$\sigma a_{fz}>0$$
    facing_side=normal[:,2]*(1 if facing=='up' else -1)>0
    def patch(extent):
        """마주 보는 경계면 중 지정한 폭 안의 삼각형을 선택해요."""
        return side[facing_side & np.all(np.abs(local[side,0])<=extent+1e-12,axis=1)
                    & np.all(np.abs(local[side,1])<=extent+1e-12,axis=1)]
    contact,query=patch(patch_half_width),patch(query_half_width)
    if len(contact)==0 or len(query)==0:
        raise ValueError('mesh too coarse for the requested facing patch; increase n_angle or n_axis')
    ends=np.r_[np.arange(width),np.arange((n_axis-1)*width,n_axis*width)]
    return dict(nodes=nodes,tetrahedra=tets,boundary_faces=boundary,
                contact_faces=contact,query_faces=query,support_nodes=ends,
                bottom_nodes=np.arange(width),top_faces=boundary[np.all(np.isclose(local[boundary,1],half_length),axis=1)],
                counts=(n_angle,n_radial,n_axis),radius=radius,half_length=half_length,
                facing=facing,angle_degrees=angle_degrees,local_nodes=local,
                side_faces=side,end_faces=boundary[end_mask],axis_nodes=axis)
